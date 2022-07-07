import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gptj.modeling_gptj import GPTJAttention as _GPTJAttention
from transformers.models.gptj.modeling_gptj import GPTJMLP as _GPTJMLP
from transformers.models.gptj.modeling_gptj import GPTJBlock as _GPTJBlock
from transformers.models.gptj.modeling_gptj import GPTJModel as _GPTJModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        
    def forward(self, input_ids, past_layer=None):
      
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)
        return hidden_states
    

class GPTBlock(_GPTJBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super().__init__(config=config, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, layer_past=None) -> torch.Tensor:
        res = x
        x = self.ln_1(x)
        x_a, present = self.attn(x, use_cache=True, layer_past=layer_past)
        x_m = self.mlp(x)
        return x_a + x_m + res, present
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
    def forward(self, x, input_ids=None):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
