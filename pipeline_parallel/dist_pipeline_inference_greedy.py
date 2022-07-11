import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.hf_gpt2_module import *


class DistInferenceAsync:
    r"""
    Async implementation of Distributed Inference.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if computation finishes in the forward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device, use_dp=False, rank=None):
        print("=======Initialize Dist Inference.")
        if rank is None:
            self.global_rank = args.rank
        else:
            self.global_rank = rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()  # Rank is the pipeline rank by default.
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()

        # assert (args.batch_size % args.micro_batch_size == 0)
        self.seq_num = args.batch_size # // args.micro_batch_size
        self.input_seq_length = args.input_seq_length
        self.generate_seq_length = args.generate_seq_length
        self.embedding_dim = args.embedding_dim
        # self.vocab_size = vocab_size
        self.num_layers = args.num_layers

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_seq_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                              for _ in range(self.seq_num)]
        self.forward_seq_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                              for _ in range(self.seq_num)]
        self.forward_token_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                                for _ in range(self.generate_seq_length)]
        self.forward_token_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                                for _ in range(self.generate_seq_length)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_seq_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.seq_num)]
            self.forward_seq_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                for _ in range(self.seq_num)]
            self.forward_token_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                    for _ in range(self.generate_seq_length)]
            if self.pp_rank == self.pipeline_group_size - 1:
                self.forward_token_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                        for _ in range(self.generate_seq_length)]
            else:
                self.forward_token_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                        for _ in range(self.generate_seq_length+1)]
            self.forward_token_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                    for _ in range(self.generate_seq_length)]
            self.forward_token_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                                  for _ in range(self.generate_seq_length)]
            self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.init_time_stamp = None

        if self.pp_rank == 0:
            self.recv_new_token = [torch.zeros((self.seq_num, 1),
                                               requires_grad=False, device=self.device, dtype=torch.int)
                                   for _ in range(self.generate_seq_length)]

        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens = [torch.zeros((self.seq_num, 1),
                                                requires_grad=False, device=self.device, dtype=torch.int)
                                    for _ in range(self.generate_seq_length)]

        self.input_seq_emb = [torch.zeros((1, self.input_seq_length, self.embedding_dim),
                                          requires_grad=False, device=self.device, dtype=torch.float32)
                              for _ in range(self.seq_num)]
        self.output_seq_emb = [torch.zeros((1, self.input_seq_length, self.embedding_dim),
                                           requires_grad=False, device=self.device, dtype=torch.float32)
                               for _ in range(self.seq_num)]
        self.input_token_emb = [torch.zeros((self.seq_num, 1, self.embedding_dim),
                                            requires_grad=False, device=self.device, dtype=torch.float32)
                                for _ in range(self.generate_seq_length)]
        self.output_token_emb = [torch.zeros((self.seq_num, 1, self.embedding_dim),
                                             requires_grad=False, device=self.device, dtype=torch.float32)
                                 for _ in range(self.generate_seq_length)]

        self.cached_attention = []
        self.prompt_input = None
        self.prompt_output = None
        self.layers = {}
        self._create_layers()

    def _create_layers(self):
        config = GPTConfig.from_pretrained('../pretrained_models/gpt2/')
        if self.pp_rank == 0:
            self.layers['emb'] = GPTEmbeddings(config).eval()
        for layer_index in range(self.num_layers):
            self.layers['block'+str(layer_index)] = GPTBlock(config).eval()
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers['lm'] = GPTLMHead(config).eval()

    def _init_cached_seqs_and_attentions(self):
        self.prompt_input = None
        self.prompt_output = None
        self.cached_attention.clear()
        for _ in range(self.num_layers-1):
            self.cached_attention.append([None for _ in range(self.seq_num)])

    def _merge_cached_seqs_and_attentions(self):
        self.prompt_input = torch.cat(self.input_seq_emb, dim=0)
        self.prompt_output = torch.cat(self.output_seq_emb, dim=0)
        for layer_index in range(self.num_layers-1):
            self.cached_attention[layer_index][0] = torch.cat(self.cached_attention[layer_index][0], dim=0)
            self.cached_attention[layer_index][1] = torch.cat(self.cached_attention[layer_index][1], dim=0)

    def _forward_compute_prompt_seq(self, index, seq=None):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers['emb'](seq)
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.cached_attention[layer_index][index] = self.layers['block' + str(layer_index)](
                    self.input_seq_emb[index])
            elif layer_index == self.num_layers - 1:
                self.output_seq_emb[index] = self.layers['block'+str(layer_index)](
                    self.cached_attention[layer_index - 1][index])
            else:
                self.cached_attention[layer_index][index] = self.layers['block' + str(layer_index)](
                    self.cached_attention[layer_index - 1][index])

    def _forward_compute_generate_token(self, step):
        print("Compute prompt seq<", step, ">.")
        if self.pp_rank == 0:
            current_emb = self.layers['emb'](self.recv_new_token[step], self.prompt_input)
        else:
            current_emb = self.input_token_emb[step]
        for layer_index in range(self.num_layers):
            if layer_index != self.num_layers - 1:
                current_emb, self.cached_attention[layer_index] = self.layers['block' + str(layer_index)](
                    current_emb, self.cached_attention[layer_index])
            else:
                self.output_token_emb[step] = self.layers['block' + str(layer_index)](
                    current_emb, self.cached_attention[layer_index])
        if self.pp_rank == self.pipeline_group_size - 1:
            self.send_new_tokens[step] = self.layers['lm'](self.output_token_emb[step])

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        self.send_new_tokens[step] = self.layers['lm'](self.output_token_emb[step])

    def profile_mark_forward_seq_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_seq_comp_start_events[i])

    def profile_mark_forward_seq_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_seq_recv_start_events[i])

    def profile_mark_forward_seq_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_seq_send_start_events[i])

    def profile_mark_forward_seq_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_seq_send_end_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_seq_pipeline_stage(self, input_data=None):
        if self.pp_rank == 0:
            assert(input_data is not None)
            input_seqs = torch.chunk(input_data, self.seq_num, dim=0)
        else:
            input_seqs = None

        for i in range(self.seq_num):
            if self.pp_rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_seq_comp_start(i)
                    self._forward_compute_prompt_seq(index=i, seq=input_seqs[i])
                    self.torch_comp_stream.record_event(self.forward_seq_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_seq_comp_ready_events[i])
                    self.profile_mark_forward_seq_send_start(i)
                    self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_seq_send_end(i)
            elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_seq_recv_start(i)
                    self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_seq_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_seq_recv_ready_events[i])
                    self.profile_mark_forward_seq_comp_start(i)
                    self._forward_compute_prompt_seq(index=i, seq=None)
                    self.torch_comp_stream.record_event(self.forward_seq_comp_ready_events[i])
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_seq_recv_start(i)
                    self.comm.recv(self.input_seq_emb[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_seq_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_seq_recv_ready_events[i])
                    self.profile_mark_forward_seq_comp_start(i)
                    self._forward_compute_prompt_seq(index=i, seq=None)
                    self.torch_comp_stream.record_event(self.forward_seq_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_seq_comp_ready_events[i])
                    self.profile_mark_forward_seq_send_start(i)
                    self.comm.send(self.output_seq_emb[i], dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_seq_send_end(i)
        if self.enable_tidy_profiling:
            self.profile_seq_pipeline_stage()

    def profile_seq_pipeline_stage(self):
        torch.cuda.synchronize()
        for i in range(self.seq_num):
            if self.pp_rank != 0:
                recv_slot = self.forward_seq_recv_start_events[i].elapsed_time(self.forward_seq_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_seq_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_seq_comp_start_events[i].elapsed_time(self.forward_seq_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_seq_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_seq_send_start_events[i].elapsed_time(self.forward_seq_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_seq_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def profile_mark_forward_token_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_token_comp_start_events[i])

    def profile_mark_forward_token_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_token_recv_start_events[i])

    def profile_mark_forward_token_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_token_send_start_events[i])

    def profile_mark_forward_token_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_token_send_end_events[i])

    def forward_new_token_pipeline_stage(self):
        self._merge_cached_seqs_and_attentions()
        if self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_comp_stream):
                self.profile_mark_forward_token_comp_start(0)
                self._generate_new_token(0)
                self.torch_comp_stream.record_event(self.forward_token_comp_ready_events[0])
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.forward_token_comp_ready_events[0])
                self.profile_mark_forward_token_send_start(0)
                self.comm.send(self.send_new_tokens[0], dst=0, stream=cupy_send_stream)
                self.profile_mark_forward_token_send_end(0)

        for i in range(self.generate_seq_length):
            if self.pp_rank == 0:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(self.recv_new_token[i], src=self.pipeline_group_size-1, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_token_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_token_recv_ready_events[i])
                    self.profile_mark_forward_token_comp_start(i)
                    self._forward_compute_generate_token(step=i)
                    self.torch_comp_stream.record_event(self.forward_token_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_token_comp_ready_events[i])
                    self.profile_mark_forward_token_send_start(i)
                    self.comm.send(self.output_token_emb[i], dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_token_send_end(i)
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_token_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_token_comp_start(i)
                    self._forward_compute_generate_token(step=i)
                    self.torch_comp_stream.record_event(self.forward_token_comp_ready_events[0])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_token_comp_ready_events[0])
                    self.profile_mark_forward_token_send_start(0)
                    self.comm.send(self.send_new_tokens[0], dst=0, stream=cupy_send_stream)
                    self.profile_mark_forward_token_send_end(0)
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_token_recv_start(i)
                    self.comm.recv(self.input_token_emb[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_token_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_token_comp_start(0)
                    self._forward_compute_generate_token(step=i)
                    self.torch_comp_stream.record_event(self.forward_token_comp_ready_events[0])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_token_comp_ready_events[0])
                    self.profile_mark_forward_token_send_start(0)
                    self.comm.send(self.output_token_emb[i], dst=0, stream=cupy_send_stream)
                    self.profile_mark_forward_token_send_end(0)

        if self.enable_tidy_profiling:
            self.profile_token_pipeline_stage()

    def profile_token_pipeline_stage(self):
        torch.cuda.synchronize()
        for i in range(self.generate_seq_length):
            if self.pp_rank != 0:
                recv_slot = self.forward_token_recv_start_events[i].elapsed_time(self.forward_token_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_token_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_token_comp_start_events[i].elapsed_time(self.forward_token_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_token_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_token_send_start_events[i].elapsed_time(self.forward_token_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_token_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)


    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def inference_batch(self, input_=None, target=None):
        pass

