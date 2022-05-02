import torch
import math
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Embeddings, ConvLayer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder as _DebertaV2Encoder
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, ContextPooler 
    

class DebertaV2Layers(_DebertaV2Encoder):
    def __init__(self, config, first_block=False):
        super(_DebertaV2Encoder, self).__init__()
        
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        if first_block:
            self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        else:
            self.conv = None
            
        self.gradient_checkpointing = True # TODO
        
        if hasattr(self, 'LayerNorm'):
            for p in self.LayerNorm.parameters():
                p.requires_grad = False
        if hasattr(self, 'rel_embeddings'):
            for p in self.rel_embeddings.parameters():
                p.requires_grad = False
        
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        next_kv = hidden_states # TODOs
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                output_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                )

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
                
            next_kv = output_states

        return output_states



class DebertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pooler = ContextPooler(config)
        self.classifier = nn.Linear(self.pooler.output_dim, getattr(config, "num_labels", 2))
        
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
    def forward(self, hidden_states, input_ids=None):
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits