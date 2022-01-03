import torch
import math
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        # in Attention: model_dim=768 (nx=n_embd)
        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.split_size = model_dim // head_num
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.scale = math.sqrt(self.split_size)

        # self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, input):
        bs = input.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(input).view(bs, -1, self.head_num, self.split_size)
        q = self.q_linear(input).view(bs, -1, self.head_num, self.split_size)
        v = self.v_linear(input).view(bs, -1, self.head_num, self.split_size)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = functional.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.model_dim)
        output = self.out(concat)
        return output + input # Put residual connection here.


class TwoLayerMLP(nn.Module):
    def __init__(self, model_dim, feedford_dim):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = nn.Linear(model_dim, feedford_dim)
        self.linear2 = nn.Linear(feedford_dim, model_dim)

    def forward(self, input):
        a1 = functional.relu(self.linear1(input))
        a2 = self.linear2(a1)
        return input + a2


class GPT3TransformerLayer(nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048, layer_norm_eps=1e-5, use_checkpoint=True) -> None:
        super(GPT3TransformerLayer, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(model_dim, head_num)
        # Implementation of Feedforward model
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2))
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.attn, x)
        else:
            x = self.attn(x)
        x = self.norm2(x)
        # x = x + self.dropout_2(self.ff(x2))
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.mlp, x)
        else:
            x = self.mlp(x)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)


class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.ntoken = ntoken
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).view(-1, self.ntoken)


class GPTShardBase(nn.Module):
    def __init__(self, args, ntokens):
        super(GPTShardBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self._num_tokens = ntokens
        self._embedding_size = args.embedding_size  # embedding dimension
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_size * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = args.num_layers

    def _create_encoder_layer(self):
        return Encoder(self._num_tokens, self._embedding_size)

    def _create_decoder_layer(self):
        return Decoder(self._num_tokens, self._embedding_size)

    def _create_transformer_layer(self):
        return GPT3TransformerLayer(self._embedding_size, self._num_heads, self._feedforward_dim,
                                    use_checkpoint=True)


class GPTShardFirst(GPTShardBase):
    def __init__(self, args, ntokens, device):
        super(GPTShardFirst, self).__init__(args, ntokens)
        self.device = device
        module_list = [self._create_encoder_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTShardMiddel(GPTShardBase):
    def __init__(self, args, ntokens, device):
        super(GPTShardMiddel, self).__init__(args, ntokens)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTShardLast(GPTShardBase):
    def __init__(self, args, ntokens, device):
        super(GPTShardLast, self).__init__(args, ntokens)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        module_list.append(self._create_decoder_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out
