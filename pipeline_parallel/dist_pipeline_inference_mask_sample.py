import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_greedy import DistGreedyInferenceMaskAsync


class DistSampleInferenceMaskAsync(DistGreedyInferenceMaskAsync):
    
    def __init__(self, args, device, rank=None):
        super().__init__(args, device, rank=rank)
        self.num_completions = args.num_completions
        self.logits_processor = get_logits_processor()
        self.logits_warper = get_logits_warper(
            top_k = (args.top_k if args.top_k > 0 else None),
            top_p = (args.top_p if 0 < args.top_p < 1 else None),
            temperature = args.temperature,
            num_beams = 1,
        )

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[step])
        z = torch.nn.functional.log_softmax(z[:, -1], -1)
        z = self.logits_warper(None, z)
        self.send_new_tokens[step] = torch.multinomial(z.softmax(-1), num_samples=1)
        
    def _init_cached_seqs_and_attentions(self):
        self.prompt_cached_attention = None
        self.cached_attention.clear()
        for _ in range(self.num_layers):
            self.cached_attention.append([None for _ in range(self.seq_num)])
        
    def _merge_cached_seqs_and_attentions(self):
        if self.prompt_cached_attention is not None:
            self.cached_attention = [
                (self.prompt_cached_attention[i][0].to(self.device),
                 self.prompt_cached_attention[i][1].to(self.device),
                ) for i in range(self.num_layers)
            ]
            return
        for layer_index in range(self.num_layers):
            key = torch.cat([kv[0] for kv in self.cached_attention[layer_index]], dim=0)
            value = torch.cat([kv[1] for kv in self.cached_attention[layer_index]], dim=0)
            self.cached_attention[layer_index] = (key, value)
        self.prompt_cached_attention = [
            (self.cached_attention[i][0].cpu(),
             self.cached_attention[i][1].cpu(),
            ) for i in range(self.num_layers)
        ]
        
    def inference_batch(self, input_=None, output_=None, attention_mask=None):
        self.comm.barrier()
        start_time = time.time()
        self._init_cached_seqs_and_attentions() # TODO: should I put here
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()

        with torch.no_grad():
            self.forward_seq_pipeline_stage(input_data=input_, attention_mask=attention_mask)
            for nc in range(self.num_completions):
                # TODO: only the first run is correct
                # the first generated token is overwritten by the second gederated token.
                self.forward_new_token_pipeline_stage(attention_mask=attention_mask)
                self.comm.barrier()
                if self.pp_rank == 0 and output_ is not None:
                    assert isinstance(output_, list)
                    output_.append(torch.cat([z.cpu() for z in self.recv_new_token], 1))

        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node whole INFERENCE iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        
        return iter_time