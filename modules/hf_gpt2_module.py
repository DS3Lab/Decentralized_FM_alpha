import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids, past_layer=None):
        if past_layer is not None:
            past_length = past_layer[0].size(2)
        else:
            past_length = 0
        device = input_ids.device
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        # position ids
        position_ids = torch.arange(
            past_length, past_length + input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        return hidden_states


class GPTBlock(_GPT2Block):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super().__init__(config=config, *args, **kargs)
        self.config = config
        self.use_checkpoint = use_checkpoint

        def attn_res(x: torch.Tensor, layer_past=None) -> torch.Tensor:
            # use_cache = (layer_past is None)
            use_cache = True
            res = x
            x = self.ln_1(x)
            x, present = self.attn(x, use_cache=use_cache, layer_past=layer_past)
            return x + res, present

        self.attn_res = attn_res

        def mlp_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x + res

        self.mlp_res = mlp_res

    def forward(self, x: torch.Tensor, layer_past=None) -> torch.Tensor:
        x, present = self.attn_res(x, layer_past=layer_past)
        x = self.mlp_res(x)
        return x, present


class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, input_ids=None):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

