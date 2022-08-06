import torch
from deepspeed.pipe import LayerSpec, PipelineModule
from .gpt_modules import GPTTransformerLayer, GPTEmbedding
from .task_modules import SeqClassification


class GlueSeqClassificationPipe(PipelineModule):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True, **kwargs):
        self.use_checkpoint = use_checkpoint

        specs = [
            LayerSpec(GPTEmbedding, vocab_size, args.embedding_dim, args.seq_length)
        ]

        for _ in range(args.num_layers):
            specs.append(LayerSpec(
                GPTTransformerLayer, args.embedding_dim, args.num_heads, args.embedding_dim*4, use_checkpoint))

        specs.append(LayerSpec(SeqClassification, args.embedding_dim, num_classes))
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)
        super().__init__(layers=specs, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)

