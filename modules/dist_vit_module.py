import numpy as np
from torch import nn
from comm.comm_utils import *

from transformers import ViTForImageClassification

from datasets import load_dataset

from copy import deepcopy


class ViTFullModel(nn.Module):
    def __init__(self, args, config=None, device='cpu'):
        super().__init__()
#         self._to_cpu = (args.dist_backend == "gloo")
#         self._embedding_dim = args.embedding_dim  # embedding dimension
#         self._seq_length = args.seq_length
#         # the dimension of the feedforward aws_network model in nn.TransformerEncoder
#         self._feedforward_dim = args.embedding_dim * 4
#         self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
#         self._num_layers = args.num_layers
#         self._layer_begin = get_pipeline_parallel_rank() * args.num_layers
#         self._layer_end = min(self._layer_begin + args.num_layers, args.max_layers)
        
#         self._task_type = getattr(args, 'task_type', 'language_model')
        
#         self.load_pretrained_model = args.load_pretrained_model
        self.model_name = args.model_name
        self.config = config
        
        ds = load_dataset(args.task_name, split='train')
        labels = ds.features['label'].names
        
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        ).to(device)
        
        
    def forward(self, x, **kargs):
        ret = self.model(x, **kargs)
        return ret.logits