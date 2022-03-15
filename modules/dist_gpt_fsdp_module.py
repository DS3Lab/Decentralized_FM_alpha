import torch
from .task_modules import GlueClassification
from .gpt_modules import MultiHeadAttention, TwoLayerMLP, GPTEmbedding, GPTTransformerLayer
from fairscale.nn.checkpoint import checkpoint_wrapper

# This is only implemented to support checkpoint in FSDP


class GPTTransformerFSDPLayer(torch.nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048, layer_norm_eps=1e-5, use_checkpoint=True) -> None:
        super(GPTTransformerFSDPLayer, self).__init__()
        self.attn = MultiHeadAttention(model_dim, head_num)
        if use_checkpoint:
            self.attn = checkpoint_wrapper(self.attn)
        # Implementation of Feedforward model
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        if use_checkpoint:
            self.mlp = checkpoint_wrapper(self.mlp)
        self.norm1 = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2))
        x.requires_grad_(True)
        x = self.attn(x)
        x = self.norm2(x)
        # x = x + self.dropout_2(self.ff(x2))
        x.requires_grad_(True)
        x = self.mlp(x)
        return x


class GPTGlueFSDPModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True):
        super(GPTGlueFSDPModel, self).__init__()
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerFSDPLayer(args.embedding_dim, args.num_heads,
                                                       args.embedding_dim*4, use_checkpoint))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = GlueClassification(args.embedding_dim, num_classes)

    def forward(self, input_ids, position_ids):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)