
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy

from .sparsification import *
from .fixpoint import *
from . import flag
    
        
class TopKCompressor:
    def __init__(
        self, ratio=0.1,
        *args, **kargs,
    ):
        self.ratio = ratio
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.activ_shape = [micro_batch_size, seq_length, embedding_dim]
        self.numel = micro_batch_size*seq_length*embedding_dim 
        self.k = max(int(self.numel * self.ratio), 1)
        
        # Communication Buffers
        self.buffers = [
            (
                torch.zeros(self.k, requires_grad=False, device=device, dtype=torch.float16),
                torch.zeros(self.numel//8, requires_grad=False, device=device, dtype=torch.uint8),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        v, m = compress_topk(x, self.k)
        return v.half(), m
        
    def decompress(self, x):
        x = decompress_topk(*x, self.activ_shape)
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            with stream:
                _data = self.compress(x)
            for _x in _data:
                comm.send(_x, dst=dst, stream=stream)
        else:
            comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            recv_buffer = self.buffers[i_micro_batch]
            for _recv_buffer in recv_buffer:
                comm.recv(_recv_buffer, src=src, stream=stream)
            with stream:
                x = self.decompress(recv_buffer)
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            return recv_buffer