import os
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
from transformers.models.bloom.modeling_bloom import BloomBlock as _BloomBlock
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from transformers.models.bloom.configuration_bloom import BloomConfig as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
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
        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = self.word_embeddings_layernorm(inputs_embeds)
        return inputs_embeds

    

class GPTBlock(_BloomBlock):
    def __init__(self, config, *args, use_checkpoint=True, device='cpu', **kargs):
        super().__init__(config=config, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        
        _reset_parameters = nn.Linear.reset_parameters
        def dummy(*args, **kargs):
            pass
        nn.Linear.reset_parameters = dummy # disable init
        module = cls(config, layer_number=layer_index).eval()
        nn.Linear.reset_parameters = _reset_parameters
        # module = torch.nn.utils.skip_init(cls, config).eval() # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, hidden_states: torch.Tensor, layer_past=None, mask=None) -> torch.Tensor:
            
        current_sequence_length = hidden_states.shape[1]
        if layer_past is not None:
            current_sequence_length += layer_past[0].shape[1]
            
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)+past_length), 
                dtype=torch.bool, device=x.device)
            
        attention_mask = mask
            
        alibi = build_alibi_tensor(current_sequence_length, self.n_head, hidden_states.dtype)
            
        layernorm_output = self.input_layernorm(hidden_states)
        
        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            use_cache=True,
        )

        attention_output = attn_outputs[0]
        present = attn_outputs[1]
        
        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)
        
        return output, present
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
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
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
