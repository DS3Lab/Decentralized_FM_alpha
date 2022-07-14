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
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention as _GPTNeoXAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP as _GPTNeoXMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer as _GPTNeoXBlock
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel as _GPTNeoXModel
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.embed_in = nn.Embedding(config.vocab_size, self.embed_dim)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print(f'Cannot load from <model_path>. The model is randomly initialized.')
        return module
        
    def forward(self, input_ids, *args, **kargs):
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embed_in(input_ids)
        return hidden_states
    

class GPTBlock(_GPTNeoXBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super().__init__(config=config, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
    
    def forward(self, x: torch.Tensor, layer_past=None, attention_mask=None) -> torch.Tensor:
        
        residual = x
        ln_out = self.input_layernorm(x)
        attention_layer_outputs = self.attention(
            ln_out,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=True,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: a, present, ...
        present = attention_layer_outputs[1]

        mlp_output = self.mlp(self.post_attention_layernorm(x))
        x = mlp_output + attn_output + residual
        
        return x, present
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
        
    def forward(self, x):
        x = self.final_layer_norm(x)
        x = self.embed_out(x)
        return x
