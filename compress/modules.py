
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy

from .fixpoint import *

class Compressor:
    def __init__(
        self, bits=4, 
        compress_method='fixpoint', 
        scale_method='max', scale_dims=(0,1),
        activ_shape=(4, 1024, 768), device='cpu',
    ):
        '''
        bits in {2, 4, 8}
        compress_method in {'fixpoint', 'delta'}
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.compress_method = compress_method
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        self.activ_shape = activ_shape
        self.scale_shape = list(activ_shape)
        for dim in scale_dims:
            self.scale_shape[dim] = 1
        self.scale_shape = tuple(self.scale_shape)
        
        num_bits_activ = self.activ_shape[0]*self.activ_shape[1]*self.activ_shape[2]*self.bits
        num_bits_scale = self.scale_shape[0]*self.scale_shape[1]*self.scale_shape[2]*16 # fp16
        assert num_bits_activ % 8 == 0
        assert num_bits_scale % 8 == 0
        self.num_comm_bytes_activ = num_bits_activ // 8
        self.num_comm_bytes_scale = num_bits_scale // 8
        activ_shape_to_comm = list(activ_shape)
        activ_shape_to_comm[-1] = activ_shape_to_comm[-1] // (8 // bits)
        self.activ_buffer = torch.zeros(activ_shape_to_comm, dtype=torch.uint8, device=device)
        self.scale_buffer = torch.zeros(self.scale_shape, dtype=torch.float16, device=device)
        self.device = device
        
    def get_comm_shape_and_dtype(self):
        return (self.activ_buffer.shape, self.activ_buffer.dtype, self.scale_buffer.shape, self.scale_buffer.dtype, )
        
    def compress(self, x):
        if self.compress_method == 'none':
            return x
        elif self.compress_method == 'fixpoint':
            if self.bits == 8:
                x, scale = compress_8bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
                return x, scale
            elif self.bits == 4:
                x, scale = compress_4bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
                return x, scale
            elif self.bits == 2:
                x, scale = compress_2bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
                return x, scale
        
        raise Exception('unknown compress method')
        
    def decompress(self, x):
        if self.compress_method == 'none':
            return x
        elif self.compress_method == 'fixpoint':
            if self.bits == 8:
                ret = decompress_8bit(*x)
                return ret
            elif self.bits == 4:
                ret = decompress_4bit(*x)
                return ret
            elif self.bits == 2:
                ret = decompress_2bit(*x)
                return ret
        
        raise Exception('unknown compress method')
            
        