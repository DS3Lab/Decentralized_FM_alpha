import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_greedy import DistGreedyInferenceAsync


class DistSampleInferenceAsync(DistGreedyInferenceAsync):

    def _generate_new_token(self, step):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[step])
        z = torch.nn.functional.log_softmax(z[:, -1], -1)
        z = self.logits_warper(None, z)
        self.send_new_tokens[step] = torch.multinomial(z.softmax(-1), num_samples=1)