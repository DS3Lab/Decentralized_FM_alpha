
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy

from .fixpoint import *


class NoCompression:
    def __init__(self, *args, **kargs):
        pass
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=torch.float32):
        self.buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        return x
        
    def decompress(self, x):
        return x
        
    def compress_send(self, x, comm, dst, stream):
        comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        comm.recv(recv_buffer, src=src, stream=stream)
        return recv_buffer
    
        
class FixpointCompressor:
    def __init__(
        self, bits=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in {2, 4, 8}
        compress_method in {'fixpoint', 'delta'}
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.buffers = [
            (
                torch.zeros((micro_batch_size, seq_length, embedding_dim * self.bits // 8), 
                            requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(scale_shape, 
                            requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        if self.bits == 8:
            x, scale = compress_8bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        elif self.bits == 4:
            x, scale = compress_4bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        elif self.bits == 2:
            x, scale = compress_2bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        
        raise Exception(f'no solution to bits {self.bits}')
        
    def decompress(self, x):
        if self.bits == 8:
            ret = decompress_8bit(*x)
            return ret
        elif self.bits == 4:
            ret = decompress_4bit(*x)
            return ret
        elif self.bits == 2:
            ret = decompress_2bit(*x)
            return ret
        
        raise Exception(f'no solution to bits {self.bits}')
        
        
    def compress_send(self, x, comm, dst, stream):
        _data = self.compress(x)
        for _x in _data:
            comm.send(_x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        for _recv_buffer in recv_buffer:
            comm.recv(_recv_buffer, src=src, stream=stream)
        x = self.decompress(recv_buffer)
        return x
            
        