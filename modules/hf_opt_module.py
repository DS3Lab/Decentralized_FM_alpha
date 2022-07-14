import os
import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.models.opt.configuration_opt import OPT2Config as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None
        
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
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, input_ids, past_layer=None):
        
        # if hasattr(self, 'tmp') and self.tmp.shape[:2] == input_ids.shape:
        #     return self.tmp
        # self.tmp = torch.zeros([*input_ids.shape, self.embed_dim], dtype=torch.float16, device=input_ids.device)
        # return self.tmp
        
        if past_layer is not None:
            past_length = past_layer[0].size(2)
        else:
            past_length = 0
        device = input_ids.device
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        inputs_embeds = self.embed_tokens(input_ids)
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        position_embeds = self.embed_positions(attention_mask, past_length)
        
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        
        hidden_states = inputs_embeds + position_embeds

        # hidden_states = self.drop(hidden_states)

        return hidden_states


class GPTBlock(OPTDecoderLayer):
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
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module

    def forward(self, x: torch.Tensor, layer_past=None) -> torch.Tensor:
        
        hidden_states = x # alias
        residual = hidden_states
        
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _, present = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=layer_past,
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        
        return hidden_states, present


class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None
            
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None
        
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        
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

    def forward(self, x, input_ids=None):
        if self.final_layer_norm is not None:
            x = self.ln_f(x)
        if self.project_out is not None:
            x = self.project_out(x)
        x = self.lm_head(x)
        return x

