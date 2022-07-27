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
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as _GPT2Attention
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as _GPT2MLP
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as _GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification as _GPT2ForSequenceClassification
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as GPTConfig

# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class DropScheduler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_count = config.layer_count
        self.pp_rank = config.pp_rank
        self.pp_size = config.pp_size
        self.batch_size = config.batch_size
        self.layer_drop_p = config.layer_drop_p
        self.layer_drop_iters = config.layer_drop_iters
        self.layer_drop_method = config.layer_drop_method
        self.dp_size = 1
        self.dp_batch_size = self.batch_size // self.dp_size
        self.micro_iter = 0
        self.global_micro_iter = 0
        self.schedule_drop()
            
    def schedule_drop(self):
        
        if self.layer_drop_method == 'none':
            return
        
        dp_size = self.dp_size
        dp_batch_size = self.dp_batch_size
        
        np.random.seed(self.global_micro_iter)
        Nk = [ # 4 dp size
            min(np.random.binomial(n=self.pp_size, p=self.layer_drop_p), 2) for _ in range(dp_size)
        ]
        self.schedule = np.zeros([self.layer_drop_iters, dp_size, dp_batch_size, self.pp_size])
        
        for j in range(len(Nk)):
#             idx_drop_layer = np.random.choice(self.pp_size, Nk[j], replace=False)
            for i in range(self.layer_drop_iters):
                for k in range(dp_batch_size):
                
                    if self.layer_drop_method == 'sample':
                        pass
                        idx_drop_layer = np.random.choice(self.pp_size, Nk[j], replace=False)
                    elif self.layer_drop_method == 'round':
                        idx_drop_layer = [z%self.pp_size for z in range(i+j+k, i+j+k+Nk[j])]
                    else:
                        raise Exception('unknown drop method')

                    self.schedule[i, j, k, idx_drop_layer] = 1
                    
        print(self.schedule.sum() / self.schedule.size)
    
    def is_drop(self):
        if not self.training:
            return False
        if self.layer_drop_method == 'none' or self.layer_drop_p == 0 or self.layer_drop_iters == 0:
            return False
        
        if self.micro_iter >= self.layer_drop_iters * self.batch_size:
            self.micro_iter = 0
            self.schedule_drop()
        
        i_macro = self.micro_iter // self.batch_size
        i_micro = (self.micro_iter % self.batch_size)
        i_pipe = i_micro // self.dp_batch_size
        i_pipe_micro = i_micro % self.dp_batch_size
        ret = self.schedule[i_macro, i_pipe, i_pipe_micro, self.pp_rank]
        
        self.global_micro_iter += 1
        self.micro_iter += 1
            
        return (ret==1)


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        
    def forward(self, input_ids):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
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
        
        def attn_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x = self.attn(x)[0]
            return x + res
        self.attn_res = attn_res
        
        def mlp_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x + res
        self.mlp_res = mlp_res
        
        def attn_res_no_drop(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x = self.attn(x)[0]
            return x / (1-config.layer_drop_p) + res
        self.attn_res_no_drop = attn_res_no_drop
        
        def mlp_res_no_drop(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x / (1-config.layer_drop_p) + res
        self.mlp_res_no_drop = mlp_res_no_drop
        
        
        self.drop_scheduler = DropScheduler(config)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if not self.training:
            x = self.attn_res(x)
            x = self.mlp_res(x)
            return x
        
        if not self.drop_scheduler.is_drop():
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.attn_res_no_drop, x)
            else:
                x = self.attn_res_no_drop(x)
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.mlp_res_no_drop, x)
            else:
                x = self.mlp_res_no_drop(x)
            return x
        else:
            config = self.config
            print(config.pp_rank * config.n_layer + config.layer_count, 'skipped!')
            return x
    
    
class GPTModel(_GPT2Model):
    def __init__(self, config):
        super(_GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size
        
        emb_layer = GPTEmbeddings(config)
        self.wte = emb_layer.wte
        self.wpe = emb_layer.wpe

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTBlock(config, layer_idx=i, use_checkpoint=True) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None, **kargs):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        hidden_states_tuple = tuple()
        for layer in self.h:
            hidden_states_tuple = hidden_states_tuple + (hidden_states,)
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        hidden_states_tuple = hidden_states_tuple + (hidden_states,)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=hidden_states_tuple,
            attentions=None,
            cross_attentions=None,
        )
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, x, input_ids=None):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    
class GPTLMHeadModel(_GPT2LMHeadModel):

    def __init__(self, config):
        super(_GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # ln_f will be calculated in self.transformer

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        # Initialize weights and apply final processing
        self.post_init()
        
class GPTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.score = nn.Linear(config.n_embd, config.num_labels, bias=False)
        
    def forward(self, hidden_states, input_ids=None):
        
        batch_size, sequence_length = hidden_states.shape[:2]
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
        
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        logits = self.score(self.ln_f(pooled_hidden_states))
        
        return logits
        
class GPTForClassification(_GPT2ForSequenceClassification):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
#     def forward(self, input_ids, labels=None):
        
#         ret = self.transformer(input_ids)
#         pool_hidden_state = ret.last_hidden_state[:, -1]
        
#         logits = self.score(pool_hidden_state)
        
#         loss = functional.cross_entropy(logits, labels)
        
#         return loss
        