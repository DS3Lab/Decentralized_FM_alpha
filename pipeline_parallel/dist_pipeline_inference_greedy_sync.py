import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


class DistGreedyInferenceSync:
    r"""
    Sync implementation of Distributed Inference.
    """

    def __init__(self, args, device, rank=None):
        print("=======Initialize Dist Inference(Sync).=======")
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16=======")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32=======")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        if rank is None:
            self.global_rank = args.rank
        else:
            self.global_rank = rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()

        self.num_layers = args.num_layers
        self.model_name = args.model_name
        self.model_type = args.model_type
        
        assert (args.batch_size % args.micro_batch_size == 0)
        self.seq_num = args.batch_size // args.micro_batch_size
        self.input_seq_length = args.input_seq_length
        self.generate_seq_length = args.generate_seq_length
        self.embedding_dim = self._get_embedding_size()
        # self.vocab_size = vocab_size

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.device = device

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_seq_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_recv_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                for _ in range(self.seq_num)]
            self.forward_seq_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_comp_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                for _ in range(self.seq_num)]
            self.forward_seq_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                for _ in range(self.seq_num)]
            self.forward_token_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                    for _ in range(self.generate_seq_length)]
            self.forward_token_recv_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.generate_seq_length)]
            self.forward_token_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                    for _ in range(self.generate_seq_length)]
            self.forward_token_comp_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.generate_seq_length)]
            self.forward_token_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                    for _ in range(self.generate_seq_length)]
            self.forward_token_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.generate_seq_length)]
            self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.init_time_stamp = None

        if self.pp_rank == 0:
            self.recv_new_token = [torch.zeros((self.seq_num, 1),
                                               requires_grad=False, device=self.device, dtype=torch.int64)
                                   for _ in range(self.generate_seq_length)]

        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens = [torch.zeros((self.seq_num, 1),
                                                requires_grad=False, device=self.device, dtype=torch.int64)
                                    for _ in range(self.generate_seq_length)]

        self.input_seq_emb = [torch.zeros((args.micro_batch_size, self.input_seq_length, self.embedding_dim),
                                          requires_grad=False, device=self.device, dtype=self.dtype)
                              for _ in range(self.seq_num)]
        self.output_seq_emb = [torch.zeros((args.micro_batch_size, self.input_seq_length, self.embedding_dim),
                                           requires_grad=False, device=self.device, dtype=self.dtype)
                               for _ in range(self.seq_num)]
        self.input_token_emb = [torch.zeros((self.seq_num, 1, self.embedding_dim),
                                            requires_grad=False, device=self.device, dtype=self.dtype)
                                for _ in range(self.generate_seq_length)]
        self.output_token_emb = [torch.zeros((self.seq_num, 1, self.embedding_dim),
                                             requires_grad=False, device=self.device, dtype=self.dtype)
                                 for _ in range(self.generate_seq_length)]

        self._print_buffers()

        self.cached_attention = []
        # self.prompt_input = None
        # self.prompt_output = None
        self.layers = {}
        self._create_layers()
        self._init_cached_seqs_and_attentions()

        self.logits_processor = get_logits_processor()  # not needed now
        self.logits_warper = get_logits_warper(
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=1,
        )

    def _print_buffers(self):
        if self.pp_rank == 0:
            if self.use_fp16:
                print("=======Rank-(0) recv_new_token: {} KB (fp16)======="
                      .format(self.seq_num * self.generate_seq_length * 2 // 1024))
            else:
                print("=======Rank-(0) recv_new_token: {} KB (fp32)======="
                      .format(self.seq_num * self.generate_seq_length * 4 // 1024))
        if self.pp_rank == self.pipeline_group_size - 1:
            if self.use_fp16:
                print("=======Rank-(N-1) send_new_token: {} KB (fp16)======="
                      .format(self.seq_num * self.generate_seq_length * 2 // 1024))
            else:
                print("=======Rank-(N-1) send_new_token: {} KB (fp32)======="
                      .format(self.seq_num * self.generate_seq_length * 4 // 1024))
        seq_emb_num = self.seq_num * self.input_seq_length * self.embedding_dim * self.seq_num
        if self.use_fp16:
            print("=======input_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(seq_emb_num * 2 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
        else:
            print("=======input_seq_emb: {} MB shape: {} X {} (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp32)======="
                  .format(seq_emb_num * 4 // 1024 // 1024, self.input_seq_emb[0].shape, self.seq_num))
        token_emb_num = self.seq_num * self.embedding_dim * self.generate_seq_length
        if self.use_fp16:
            print("=======input_token_emb: {} MB shape: {} X {} (fp16)======="
                  .format(token_emb_num * 2 // 1024 // 1024, self.input_token_emb[0].shape, self.generate_seq_length))
            print("=======output_seq_emb: {} MB shape: {} X {} (fp16)======="
                  .format(token_emb_num * 2 // 1024 // 1024, self.output_token_emb[0].shape, self.generate_seq_length))
        else:
            print("=======input_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(token_emb_num * 4 // 1024 // 1024, self.input_token_emb[0].shape, self.generate_seq_length))
            print("=======output_token_emb: {} MB shape: {} X {} (fp32)======="
                  .format(token_emb_num * 4 // 1024 // 1024, self.output_token_emb[0].shape, self.generate_seq_length))
            
    def _get_embedding_size(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        else:
            raise Exception(f'unknown model type {self.model_type}')

    def _create_layers(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        else:
            raise Exception(f'unknown model type {self.model_type}')

        if self.pp_rank == 0:
            self.layers['emb'] = GPTEmbeddings.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
        for layer_index in range(self.num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.num_layers * self.pp_rank + layer_index
            print(f'loading layer {global_layer_index}')
            self.layers['block' + str(layer_index)] = GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(self.dtype).eval().to(self.device)
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers['lm'] = GPTLMHead.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)

    def _init_cached_seqs_and_attentions(self):
        # self.prompt_input = None
        # self.prompt_output = None
        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])

    def _merge_cached_seqs_and_attentions(self):
        # self.prompt_input = torch.cat(self.input_seq_emb, dim=0) # TODO
        # self.prompt_output = torch.cat(self.output_seq_emb, dim=0) # TODO
        for layer_index in range(self.num_layers):
            key = torch.cat([kv[0] for kv in self.cached_attention[layer_index]], dim=0)
            value = torch.cat([kv[1] for kv in self.cached_attention[layer_index]], dim=0)
            self.cached_attention[layer_index] = (key, value)
            if self.use_fp16:
                print("=======Layer {} cached key: {} MB shape: {} (fp16)======="
                      .format(layer_index, torch.numel(key) * 2 // 1024 // 1024, key.shape))
                print("=======Layer {} cached key: {} MB shape: {} (fp16)======="
                      .format(layer_index, torch.numel(value) * 2 // 1024 // 1024, value.shape))
            else:
                print("=======Layer {} cached key: {} MB shape: {} (fp32)======="
                      .format(layer_index, torch.numel(key) * 4 // 1024 // 1024, key.shape))
                print("=======Layer {} cached key: {} MB shape: {} (fp32)======="
                      .format(layer_index, torch.numel(value) * 4 // 1024 // 1024, value.shape))

    def _forward_compute_prompt_seq(self, index, seq=None):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers['emb'](seq)
        current_emb = None
        with torch.no_grad():
            for layer_index in range(self.num_layers):
                if layer_index == 0:
                    current_emb, self.cached_attention[layer_index][index] = \
                        self.layers['block' + str(layer_index)](self.input_seq_emb[index])
                elif layer_index == self.num_layers - 1:
                    self.output_seq_emb[index], self.cached_attention[layer_index][index] = \
                        self.layers['block' + str(layer_index)](current_emb)
                else:
                    current_emb, self.cached_attention[layer_index][index] = \
                        self.layers['block' + str(layer_index)](current_emb)
        if self.pp_rank == self.pipeline_group_size - 1:
            self.output_token_emb[0][index] = current_emb[:, -1:]

    def _forward_compute_generate_token(self, step):
        print("Compute generate seq<", step, ">.")
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[step], self.cached_attention[0])
        else:
            current_emb = self.input_token_emb[step]
        with torch.no_grad():
            for layer_index in range(self.num_layers):
                if layer_index != self.num_layers - 1:
                    current_emb, self.cached_attention[layer_index] = self.layers['block' + str(layer_index)](
                        current_emb, self.cached_attention[layer_index])
                else:
                    self.output_token_emb[step], self.cached_attention[layer_index] = self.layers[
                        'block' + str(layer_index)](
                        current_emb, self.cached_attention[layer_index])
        if self.pp_rank == self.pipeline_group_size - 1:
            self._generate_new_token(step)

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[step])
        self.send_new_tokens[step] = z.argmax(-1)
        # print(step, self.send_new_tokens[step][0])

    def profile_mark_forward_seq_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_recv_start_events[i].record()

    def profile_mark_forward_seq_recv_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_recv_end_events[i].record()

    def profile_mark_forward_seq_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_comp_start_events[i].record()

    def profile_mark_forward_seq_comp_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_comp_end_events[i].record()

    def profile_mark_forward_seq_send_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_send_start_events[i].record()

    def profile_mark_forward_seq_send_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_seq_send_end_events[i].record()

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_seq_pipeline_stage(self, input_data=None):
        if self.pp_rank == 0:
            assert (input_data is not None)
            input_seqs = torch.chunk(input_data, self.seq_num, dim=0)
        else:
            input_seqs = None
            
        for i in range(self.seq_num):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                # Compute
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(index=i, seq=input_seqs[i])
                self.profile_mark_forward_seq_comp_end(i)
                # Send
                self.profile_mark_forward_seq_send_start(i)
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_seq_send_end(i)
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                # Receive
                self.profile_mark_forward_seq_recv_start(i)
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_seq_recv_end(i)
                # Compute
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(index=i, seq=None)
                self.profile_mark_forward_seq_comp_end(i)
            else:  # receive, compute, and send
                # Receive
                self.profile_mark_forward_seq_recv_start(i)
                self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_seq_recv_end(i)
                # Compute
                self.profile_mark_forward_seq_comp_start(i)
                self._forward_compute_prompt_seq(index=i, seq=None)
                self.profile_mark_forward_seq_comp_end(i)
                # Send
                self.profile_mark_forward_seq_send_start(i)
                self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_seq_send_end(i)
        if self.enable_tidy_profiling:
            self.profile_seq_pipeline_stage()

    def profile_seq_pipeline_stage(self):
        torch.cuda.synchronize()
        for i in range(self.seq_num):
            if self.pp_rank != 0:
                recv_slot = self.forward_seq_recv_start_events[i].elapsed_time(
                    self.forward_seq_recv_end_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_seq_recv_start_events[i]), "dur": recv_slot,
                            "args": {"seq-index": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_seq_comp_start_events[i].elapsed_time(self.forward_seq_comp_end_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_seq_comp_start_events[i]), "dur": comp_slot,
                        "args": {"seq-index": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_seq_send_start_events[i].elapsed_time(
                    self.forward_seq_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_seq_send_start_events[i]), "dur": send_slot,
                            "args": {"seq-index": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def profile_mark_forward_token_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_recv_start_events[i].record()

    def profile_mark_forward_token_recv_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_recv_end_events[i].record()

    def profile_mark_forward_token_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_comp_start_events[i].record()

    def profile_mark_forward_token_comp_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_comp_end_events[i].record()

    def profile_mark_forward_token_send_start(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_send_start_events[i].record()

    def profile_mark_forward_token_send_end(self, i):
        if self.enable_tidy_profiling:
            self.forward_token_send_end_events[i].record()

    def forward_new_token_pipeline_stage(self):
        self._merge_cached_seqs_and_attentions()
        if self.pp_rank == self.pipeline_group_size - 1:
            # Compute
            self.profile_mark_forward_token_comp_start(0)
            self._generate_new_token(0)
            self.profile_mark_forward_token_comp_end(0)
            # Send
            self.profile_mark_forward_token_send_start(0)
            self.comm.send(self.send_new_tokens[0], dst=0)
            self.profile_mark_forward_token_send_end(0)
            
        for i in range(self.generate_seq_length):
                
            if self.pp_rank == 0:
                # Receive
                self.profile_mark_forward_token_recv_start(i)
                self.comm.recv(self.recv_new_token[i], src=self.pipeline_group_size - 1)
                self.profile_mark_forward_token_recv_end(i)
                # Compute
                self.profile_mark_forward_token_comp_start(i)
                self._forward_compute_generate_token(step=i)
                self.profile_mark_forward_token_comp_end(i)
                # Send
                self.profile_mark_forward_token_send_start(i)
                self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_token_send_end(i)
            elif self.pp_rank == self.pipeline_group_size - 1:
                # Receive
                self.profile_mark_forward_token_recv_start(i)
                self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_token_recv_end(i)
                if i != self.generate_seq_length - 1:
                    # Compute
                    self.profile_mark_forward_token_comp_start(i + 1)
                    self._forward_compute_generate_token(step=i)  # Note: i+1 is wrong. tiny up tomorrow
                    self.profile_mark_forward_token_comp_end(i + 1)
                    # Send
                    self.profile_mark_forward_token_send_start(i + 1)
                    self.comm.send(self.send_new_tokens[i], dst=0)  # Note: i+1 is wrong. tiny up tomorrow
                    self.profile_mark_forward_token_send_end(i + 1)
            else:
                # Recv
                self.profile_mark_forward_token_recv_start(i)
                self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank)
                self.profile_mark_forward_token_recv_end(i)
                # Compute
                self.profile_mark_forward_token_comp_start(i)
                self._forward_compute_generate_token(step=i)
                self.profile_mark_forward_token_comp_end(i)
                # Send
                self.profile_mark_forward_token_send_start(i)
                self.comm.send(self.output_token_emb[i], dst=self.post_node_rank)
                self.profile_mark_forward_token_send_end(i)

        if self.enable_tidy_profiling:
            self.profile_token_pipeline_stage()

    def profile_token_pipeline_stage(self):
        torch.cuda.synchronize()
        for i in range(self.generate_seq_length):
            if self.pp_rank != 0:
                recv_slot = self.forward_token_recv_start_events[i].elapsed_time(
                    self.forward_token_recv_end_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_token_recv_start_events[i]), "dur": recv_slot,
                            "args": {"token-step": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_token_comp_start_events[i].elapsed_time(
                self.forward_token_comp_end_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_token_comp_start_events[i]), "dur": comp_slot,
                        "args": {"token-step": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_token_send_start_events[i].elapsed_time(
                    self.forward_token_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_token_send_start_events[i]), "dur": send_slot,
                            "args": {"token-step": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def inference_batch(self, input_=None, output_=None, **kargs):
        self._init_cached_seqs_and_attentions()  # TODO: should I put here
        self.comm.barrier()
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()

        self.forward_seq_pipeline_stage(input_data=input_)
        self.comm.barrier()
        prompt_time = time.time()
        print("Rank {} node INFERENCE prompt takes {:3.2f}s".format(self.global_rank, prompt_time - start_time))

        self.forward_new_token_pipeline_stage()

        self.comm.barrier()
        if self.pp_rank == 0 and output_ is not None:
            assert isinstance(output_, list)
            item = {}
            if self.generate_seq_length > 0:
                item = {
                    'token_ids': torch.cat([z.cpu() for z in self.recv_new_token], 1),
                }
            output_.append(item)
        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node INFERENCE new token takes {:3.2f}s".format(self.global_rank, end_time - prompt_time))
        print("Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        
        return iter_time

