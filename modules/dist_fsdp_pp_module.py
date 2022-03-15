import torch
from .task_modules import GlueClassification
from .gpt_modules import GPTEmbedding, GPTTransformerLayer
from fairscale.nn.checkpoint import checkpoint_wrapper


class GPTGlueFSDPModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes):
        super(GPTGlueFSDPModel, self).__init__()
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(checkpoint_wrapper(GPTTransformerLayer(args.embedding_dim, args.num_heads,
                                                                      args.embedding_dim*4, use_checkpoint=False)))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = GlueClassification(args.embedding_dim, num_classes)

    def forward(self, input_ids, position_ids):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)