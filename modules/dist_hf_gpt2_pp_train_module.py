import numpy as np
from torch import nn
from .hf_gpt2_train_module import GPTEmbeddings, GPTBlock, GPTClassificationHead, GPTLMHead

from copy import deepcopy


class GPTStageBase(nn.Module):
    def __init__(self, args, config):
        super(GPTStageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = args.num_layers
        self._task_type = getattr(args, 'task_type', 'classification')
        self._layer_count = 0
        
        self.config = config

    def _create_first_layer(self):
        return GPTEmbeddings(deepcopy(self.config))

    def _create_last_layer(self):
        if self._task_type == 'classification':
            return GPTClassificationHead(deepcopy(self.config))
        elif self._task_type == 'language_model':
            return GPTLMHead(deepcopy(self.config))
        raise Exception('unknown data type')

    def _create_transformer_layer(self):
        config = deepcopy(self.config)
        config.layer_count = self._layer_count
        self._layer_count += 1
        return GPTBlock(config) # TODO: checkpoint


class GPTStageFirst(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageFirst, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageMiddle, self).__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageLast, self).__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x, input_ids=None):
        if input_ids is None:
            out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        else:
            x = x.to(self.device) if self._to_cpu else x
            input_ids = input_ids.to(self.device) if self._to_cpu else input_ids
            for layer in self.model[:-1]:
                x = layer(x)
            out = self.model[-1](x, input_ids=input_ids)
            
        return out.cpu() if self._to_cpu else out