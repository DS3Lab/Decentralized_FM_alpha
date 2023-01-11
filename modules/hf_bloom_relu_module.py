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


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    batch_size, source_length = mask.size()
    # TODO: do not expand
    # tgt_len = tgt_len if tgt_len is not None else source_length
    # expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, source_length).to(dtype)
    expanded_mask = mask[:, None, None, :].to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


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
        
        n_head = self.n_head
        closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
        base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != n_head:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
            )
            num_remaining_heads = min(closest_power_of_2, n_head - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        self.slopes = slopes
        
        dtype = torch.float32
        mask = torch.full((3000, 3000), torch.finfo(dtype).min)
        mask_cond = torch.arange(3000)
        intermediate_mask = mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
        mask.masked_fill_(intermediate_mask, 0)
        # self.cache_mask = mask
        self.register_buffer('cache_mask', mask, persistent=False)
        
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
            ), map_location='cpu'))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')

        module.layer_index = layer_index
        return module
    
    def _make_causal_mask(
        self, input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        batch_size, target_length = input_ids_shape
        mask = self.cache_mask[:target_length, :target_length].to(dtype)
        
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(target_length, past_key_values_length, dtype=dtype, device=mask.device), mask], dim=-1)
            
        return mask[None, None, :, :]
        # expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
        # return expanded_mask
    
    def _prepare_attn_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(attention_mask.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def _build_alibi_tensor(self, attention_mask: torch.Tensor, n_head: int, dtype, device) -> torch.Tensor:
        if self.slopes.device != attention_mask.device:
            self.slopes = self.slopes.to(attention_mask.device)
        slopes = self.slopes
        arange_tensor = (attention_mask.cumsum(-1)[:, None, :] - 1) * attention_mask[:, None]
        alibi = slopes.unsqueeze(-1) * arange_tensor
        alibi = alibi * attention_mask[:, None]
        return alibi.reshape(alibi.shape[0] * n_head, 1, -1).to(dtype)

    def forward(self, hidden_states: torch.Tensor, layer_past=None, mask=None) -> torch.Tensor:
            
        current_sequence_length = hidden_states.shape[1]
        past_key_values_length = 0
        if layer_past is not None:
            # permute to be compatible
            # (bs, nhead, seq, size) => (bs, seq, nhead, size)
            layer_past = (layer_past[0].permute(0, 2, 1, 3), layer_past[1].permute(0, 2, 1, 3))
            past_key_values_length = layer_past[0].shape[1]
            current_sequence_length += past_key_values_length
            
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)+past_length), 
                dtype=torch.bool, device=x.device)
            
        attention_mask = mask
            
        # alibi = build_alibi_tensor(current_sequence_length, self.n_head, hidden_states.dtype)
        alibi = self._build_alibi_tensor(attention_mask, self.n_head, hidden_states.dtype, hidden_states.device)
        input_shape = hidden_states.size()[:-1]
        causal_mask = self._prepare_attn_mask(attention_mask, input_shape, hidden_states, past_key_values_length)
            
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
            attention_mask=causal_mask,
            alibi=alibi,
            use_cache=True,
        )

        attention_output = attn_outputs[0]
        present = attn_outputs[1]
        # permute to be compatible
        # (bs, seq, nhead, size) => (bs, nhead, seq, size)
        present = (present[0].permute(0, 2, 1, 3), present[1].permute(0, 2, 1, 3))
        
        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        #output = self.mlp(layernorm_output, residual)
        hidden_states = layernorm_output.view(-1, layernorm_output.size(-1))
        hidden_states = self.mlp.dense_h_to_4h(hidden_states)

        # p_topk = 0.1
        # _, indices = hidden_states.topk(k=int(p_topk*hidden_states.size(-1)), dim=-1)
        # zeros = torch.zeros_like(hidden_states, dtype=torch.bool)
        # mask_top = zeros.scatter(-1, indices, True)
        # mask_pos = (hidden_states > 0)

        hidden_states = torch.nn.functional.gelu(hidden_states)
        #hidden_states = self.mlp.gelu_impl(hidden_states)

        # original_sum = torch.mean(torch.abs(hidden_states), dim=-1, keepdim=True)

        p_bottom = 1.0
        _, indices = torch.abs(hidden_states).topk(k=int(p_bottom*hidden_states.size(-1)), dim=-1)
        zeros = torch.zeros_like(hidden_states, dtype=torch.bool)
        mask_bottom = zeros.scatter(-1, indices, True)

        hidden_states[~mask_bottom] = 0
        # current_sum = torch.mean(torch.abs(hidden_states), dim=-1, keepdim=True)
        # scale = original_sum / (current_sum + 1e-6)
        # hidden_states *= scale
        
        output = self.mlp.dense_4h_to_h(hidden_states).view(residual.shape) + residual
        
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
