
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy
from torch.utils.dlpack import to_dlpack, from_dlpack
import concurrent.futures
import tempfile
import time

from .fixpoint import *
from .sparsification import *
from .utils import *
from . import flag


MAX_CACHE_SIZE = 3000


class DeltaCompressor:
    def __init__(
        self, bits=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in [1, 8]
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future_read = None
        self.future_write = None
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.scale_shape = scale_shape
        
        # Activation Cache
        self.tmp_f = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache = np.memmap(
            self.tmp_f, mode='w+', dtype=np.float16, shape=(MAX_CACHE_SIZE, 2, seq_length, embedding_dim),
        )
        # Info: ensure it has content, so the profiling will be accurate
        #self.cache.fill(0)
        
        # Communication Buffers
        _x = torch.randn(self.activ_shape).to(device)
        _a, _s = compress_flexible_nbits(_x, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        self.buffers = [
            (
                torch.zeros(_a.shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(_s.shape, requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # CPU RAM Buffers
        self.np_dec_buffers = [
            pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_buffers = [
            pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        
        # GPU RAM Buffers
        self.cp_dec_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        
    def _read_from_cache(self, sample_ids):
#         time.sleep(1)
        activations = self.cache[sample_ids]
        a_dec, a_com = activations[:, 0], activations[:, 1]
        for i in range(self.batch_size//self.micro_batch_size):
            self.np_dec_buffers[i][:] = a_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_buffers[i][:] = a_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            
    def _write_to_cache(self, sample_ids):
        dec_activations = np.concatenate(self.np_dec_buffers, 0)
        com_activations = np.concatenate(self.np_com_buffers, 0)
        activations = np.stack([dec_activations, com_activations], 1)
        self.cache[sample_ids] = activations
        self.cache.flush()
        
    def _wait_read(self):
        if self.future_read is not None:
            self.future_read.result()
            self.future_read = None
            
    def _wait_write(self):
        if self.future_write is not None:
            self.future_write.result()
            self.future_write = None
            
    def read_from_cache(self, sample_ids):
        self._wait_read()
        self._wait_write()
        self.future_read = self.executor.submit(self._read_from_cache, sample_ids)
        
    def write_to_cache(self, sample_ids):
        self._wait_read()
        self._wait_write()
        self.future_write = self.executor.submit(self._write_to_cache, sample_ids)
        
    def write_and_read_cache(self, write_ids, read_ids):
        self._wait_read()
        self._wait_write()
        self.future_write = self.executor.submit(self._write_to_cache, write_ids)
        self.future_read = self.executor.submit(self._read_from_cache, read_ids)
        
    def compress(self, x, i_micro_batch):
        # get cache
        self.cp_com_buffers[i_micro_batch].set(self.np_com_buffers[i_micro_batch])
        last_x = cupy_to_tensor(self.cp_com_buffers[i_micro_batch])
        delta = x - last_x
        # compresss delta
        compressed_delta = compress_flexible_nbits(delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        # update cache
        delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
        x = last_x + delta
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_buffers[i_micro_batch].set(self.np_dec_buffers[i_micro_batch])
        last_x = cupy_to_tensor(self.cp_dec_buffers[i_micro_batch])
        # decompress delta
        delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
        # update cache
        x = last_x + delta
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_dec_buffers[i_micro_batch])
        return x
    
    def no_compress(self, x, i_micro_batch):
        # update cache
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return x
        
    def no_decompress(self, x, i_micro_batch):
        # update cache
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_dec_buffers[i_micro_batch])
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        self._wait_read()
        self._wait_write()
        
        if not flag.FLAG_DISABLE_COMPRESSION:
            with stream:
                _data = self.compress(x, i_micro_batch=i_micro_batch)
            for _x in _data:
                comm.send(_x, dst=dst, stream=stream)
        else:
            with stream:
                x = self.no_compress(x, i_micro_batch=i_micro_batch)
            comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        self._wait_read()
        self._wait_write()
        if not flag.FLAG_DISABLE_COMPRESSION:
            recv_buffer = self.buffers[i_micro_batch]
            for _recv_buffer in recv_buffer:
                comm.recv(_recv_buffer, src=src, stream=stream)
            with stream:
                x = self.decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            with stream:
                x = self.no_decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        
        
        
class DeltaLowBitsCompressor(DeltaCompressor):
    def __init__(
        self, bits=4, bits_act=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in [1, 8]
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.bits_act = bits_act
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future_read = None
        self.future_write = None
        
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.scale_shape = scale_shape
        
        # Activation Cache
        _x = torch.randn((seq_length, embedding_dim)).to(device)
        # Note: cannot use 'compress_flexible_nbits' as it will pack the entire micro batch
        _a, _s = compress_nbits(
            _x, self.bits_act, scale_method='max', scale_dims=(0,))
        self.compressed_activ_shape = _a.shape
        self.compressed_scale_shape = _s.shape
        self.tmp_f = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_activ = np.memmap(
            self.tmp_f, mode='w+', dtype=np.uint8, shape=(
                MAX_CACHE_SIZE, 2, *self.compressed_activ_shape,
            ),
        )
        self.tmp_f2 = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_scale = np.memmap(
            self.tmp_f2, mode='w+', dtype=np.float16, shape=(
                MAX_CACHE_SIZE, 2, *self.compressed_scale_shape,
            ),
        )
        
        # Communication Buffers
        _x = torch.randn(self.activ_shape).to(device)
        _a, _s = compress_flexible_nbits(
            _x, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        self.buffers = [
            (
                torch.zeros(_a.shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(_s.shape, requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # CPU RAM Buffers
        self.micro_compressed_activ_shape = (self.micro_batch_size, *self.compressed_activ_shape)
        self.micro_compressed_scale_shape = (self.micro_batch_size, *self.compressed_scale_shape)
        self.np_dec_activ_buffers = [
            pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_dec_scale_buffers = [
            pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_activ_buffers = [
            pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_scale_buffers = [
            pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        
        # GPU RAM Buffers
        self.cp_dec_activ_buffers = [
            cupy.empty(self.micro_compressed_activ_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_dec_scale_buffers = [
            cupy.empty(self.micro_compressed_scale_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_activ_buffers = [
            cupy.empty(self.micro_compressed_activ_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_scale_buffers = [
            cupy.empty(self.micro_compressed_scale_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        
    def _read_from_cache(self, sample_ids):
        activations = self.cache_activ[sample_ids]
        scales = self.cache_scale[sample_ids]
        a_dec, a_com = activations[:, 0], activations[:, 1]
        s_dec, s_com = scales[:, 0], scales[:, 1]
        for i in range(self.batch_size//self.micro_batch_size):
            self.np_dec_activ_buffers[i][:] = a_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_activ_buffers[i][:] = a_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_dec_scale_buffers[i][:] = s_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_scale_buffers[i][:] = s_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            
    def _write_to_cache(self, sample_ids):
        dec_activations = np.concatenate(self.np_dec_activ_buffers, 0)
        com_activations = np.concatenate(self.np_com_activ_buffers, 0)
        dec_scales = np.concatenate(self.np_dec_scale_buffers, 0)
        com_scales = np.concatenate(self.np_com_scale_buffers, 0)
        activations = np.stack([dec_activations, com_activations], 1)
        scales = np.stack([dec_scales, com_scales], 1)
        self.cache_activ[sample_ids] = activations
        self.cache_scale[sample_ids] = scales
        
    def compress(self, x, i_micro_batch):
        # get cache
        self.cp_com_activ_buffers[i_micro_batch].set(self.np_com_activ_buffers[i_micro_batch])
        self.cp_com_scale_buffers[i_micro_batch].set(self.np_com_scale_buffers[i_micro_batch])
        last_compressed_activ = cupy_to_tensor(self.cp_com_activ_buffers[i_micro_batch])
        last_compressed_scale = cupy_to_tensor(self.cp_com_scale_buffers[i_micro_batch])
        last_x = decompress_nbits(
            last_compressed_activ, last_compressed_scale, 
            bits=self.bits_act)
        # compresss delta
        delta = x - last_x
        compressed_delta = compress_flexible_nbits(
            delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        # update cache
        delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
        x = last_x + delta
#         sparse_x = x.clone()
#         _, masks = topr(sparse_x, 0.2)
#         sparse_x[~masks] = 0
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = tensor_to_cupy(compressed_x[0])
        s_cp = tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_activ_buffers[i_micro_batch].set(self.np_dec_activ_buffers[i_micro_batch])
        self.cp_dec_scale_buffers[i_micro_batch].set(self.np_dec_scale_buffers[i_micro_batch])
        last_compressed_activ = cupy_to_tensor(self.cp_dec_activ_buffers[i_micro_batch])
        last_compressed_scale = cupy_to_tensor(self.cp_dec_scale_buffers[i_micro_batch])
        last_x = decompress_nbits(
            last_compressed_activ, last_compressed_scale, 
            bits=self.bits_act)
        # decompress delta
        delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
        # update cache
        x = last_x + delta
#         sparse_x = x.clone()
#         _, masks = topr(sparse_x, 0.2)
#         sparse_x[~masks] = 0
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = tensor_to_cupy(compressed_x[0])
        s_cp = tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x
    
    def no_compress(self, x, i_micro_batch):
        # update cache
#         sparse_x = x.clone()
#         _, masks = topr(sparse_x, 0.2)
#         sparse_x[~masks] = 0
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = tensor_to_cupy(compressed_x[0])
        s_cp = tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return x
        
    def no_decompress(self, x, i_micro_batch):
        # update cache
#         sparse_x = x.clone()
#         _, masks = topr(sparse_x, 0.2)
#         sparse_x[~masks] = 0
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = tensor_to_cupy(compressed_x[0])
        s_cp = tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x
    

    
from .fixpoint import _compress_nbits, _decompress_nbits
    
def _compress_topk_nbits(x, k, bits, scale_method='max'):
    batch_size = x.size(0)
    
    # 1. sparsify x (keep batch dims)
    x_flat = x.view(batch_size, -1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.bool)
    for i in range(batch_size):
        masks[i, indexes[i]] = 1
    masks = masks.view(x.shape)
    x = x * masks
    
    # 2. quantize x        
    uint_x, scales = _compress_nbits(x, bits, scale_method=scale_method, scale_dims=(1,))
    topk_uint_x = uint_x[masks].view(-1, k)
    if bits == 8:
        pass
    elif bits == 4:
        x0, x1 = topk_uint_x.chunk(2, -1)
        topk_uint_x = (x0 << 4) + x1
    elif bits == 2:
        x0, x1, x2, x3 = topk_uint_x.chunk(4, -1)
        topk_uint_x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3
    else:
        raise Exception('not support bits')
    masks = cupy_to_tensor(
        cupy.packbits(tensor_to_cupy(masks))
    ).view(batch_size, -1)
    return topk_uint_x, masks, scales

def _decompress_topk_nbits(topk_uint_x, masks, scales, k, bits, original_shape):
    masks = cupy_to_tensor(
        cupy.unpackbits(tensor_to_cupy(masks))
    )
    masks = masks.view(original_shape)
    if bits == 8:
        pass
    elif bits == 4:
        bitmask = 15
        x0 = (topk_uint_x >> 4)
        x1 = (topk_uint_x & bitmask)
        topk_uint_x = torch.cat([x0, x1], -1)
    elif bits == 2:
        bitmask = 3
        x0 = (topk_uint_x >> 6)
        x1 = (topk_uint_x >> 4) & bitmask
        x2 = (topk_uint_x >> 2) & bitmask
        x3 = topk_uint_x & bitmask
        topk_uint_x = torch.cat([x0, x1, x2, x3], -1)
    else:
        raise Exception('not support bits')
    uint_x = torch.zeros(original_shape, dtype=torch.uint8, device=topk_uint_x.device) + (1<<bits-1)
    uint_x[masks] = topk_uint_x[masks.any(-1).any(-1)].view(-1)
    x = _decompress_nbits(uint_x, scales, bits)
    return x
    
    
class DeltaTopKLowBitsCompressor(DeltaCompressor):
    def __init__(
        self, bits=4, bits_act=4, ratio_act=0.1,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in [1, 8]
        bits_act in {2, 4, 8}
        ratio in [0, 1]
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.bits_act = bits_act
        self.ratio_act = ratio_act
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future_read = None
        self.future_write = None
        
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        self.k_act = int(seq_length * embedding_dim * self.ratio_act)
        self.k_act = self.k_act + ((8//self.bits_act) - self.k_act % (8//self.bits_act))
        self.k_act = max(self.k_act, 1)
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.scale_shape = scale_shape
        
        # Activation Cache
        self.compressed_activ_shape = (self.k_act*self.bits_act//8,)
        self.compressed_masks_shape = (seq_length*embedding_dim//8,)
        self.compressed_scale_shape = (1,embedding_dim,)
        self.tmp_f = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_activ = np.memmap(
            self.tmp_f, mode='w+', dtype=np.uint8, shape=(
                MAX_CACHE_SIZE, 2, *self.compressed_activ_shape,
            ),
        )
        self.tmp_f2 = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_masks = np.memmap(
            self.tmp_f2, mode='w+', dtype=np.uint8, shape=(
                MAX_CACHE_SIZE, 2, *self.compressed_masks_shape,
            ),
        )
        self.tmp_f3 = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_scale = np.memmap(
            self.tmp_f3, mode='w+', dtype=np.float16, shape=(
                MAX_CACHE_SIZE, 2, *self.compressed_scale_shape,
            ),
        )
        
        # Communication Buffers
        _x = torch.randn(self.activ_shape).to(device)
        _a, _s = compress_flexible_nbits(
            _x, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        self.buffers = [
            (
                torch.zeros(_a.shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(_s.shape, requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # CPU RAM Buffers
        self.micro_compressed_activ_shape = (self.micro_batch_size, *self.compressed_activ_shape)
        self.micro_compressed_masks_shape = (self.micro_batch_size, *self.compressed_masks_shape)
        self.micro_compressed_scale_shape = (self.micro_batch_size, *self.compressed_scale_shape)
        self.np_dec_activ_buffers = [
            pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_dec_masks_buffers = [
            pin_memory(np.zeros(self.micro_compressed_masks_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_dec_scale_buffers = [
            pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_activ_buffers = [
            pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_masks_buffers = [
            pin_memory(np.zeros(self.micro_compressed_masks_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_scale_buffers = [
            pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        
        # GPU RAM Buffers
        self.cp_dec_activ_buffers = [
            cupy.empty(self.micro_compressed_activ_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_dec_masks_buffers = [
            cupy.empty(self.micro_compressed_masks_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_dec_scale_buffers = [
            cupy.empty(self.micro_compressed_scale_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_activ_buffers = [
            cupy.empty(self.micro_compressed_activ_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_masks_buffers = [
            cupy.empty(self.micro_compressed_masks_shape, dtype=np.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_scale_buffers = [
            cupy.empty(self.micro_compressed_scale_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        
    def _read_from_cache(self, sample_ids):
        activations = self.cache_activ[sample_ids]
        masks = self.cache_masks[sample_ids]
        scales = self.cache_scale[sample_ids]
        a_dec, a_com = activations[:, 0], activations[:, 1]
        m_dec, m_com = masks[:, 0], masks[:, 1]
        s_dec, s_com = scales[:, 0], scales[:, 1]
        for i in range(self.batch_size//self.micro_batch_size):
            self.np_dec_activ_buffers[i][:] = a_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_activ_buffers[i][:] = a_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_dec_masks_buffers[i][:] = m_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_masks_buffers[i][:] = m_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_dec_scale_buffers[i][:] = s_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            self.np_com_scale_buffers[i][:] = s_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            
    def _write_to_cache(self, sample_ids):
        dec_activations = np.concatenate(self.np_dec_activ_buffers, 0)
        com_activations = np.concatenate(self.np_com_activ_buffers, 0)
        dec_masks = np.concatenate(self.np_dec_masks_buffers, 0)
        com_masks = np.concatenate(self.np_com_masks_buffers, 0)
        dec_scales = np.concatenate(self.np_dec_scale_buffers, 0)
        com_scales = np.concatenate(self.np_com_scale_buffers, 0)
        activations = np.stack([dec_activations, com_activations], 1)
        masks = np.stack([dec_masks, com_masks], 1)
        scales = np.stack([dec_scales, com_scales], 1)
        self.cache_activ[sample_ids] = activations
        self.cache_masks[sample_ids] = masks
        self.cache_scale[sample_ids] = scales
        
    def compress(self, x, i_micro_batch):
        # get cache
        self.cp_com_activ_buffers[i_micro_batch].set(self.np_com_activ_buffers[i_micro_batch])
        self.cp_com_masks_buffers[i_micro_batch].set(self.np_com_masks_buffers[i_micro_batch])
        self.cp_com_scale_buffers[i_micro_batch].set(self.np_com_scale_buffers[i_micro_batch])
        last_compressed_activ = cupy_to_tensor(self.cp_com_activ_buffers[i_micro_batch])
        last_compressed_masks = cupy_to_tensor(self.cp_com_masks_buffers[i_micro_batch])
        last_compressed_scale = cupy_to_tensor(self.cp_com_scale_buffers[i_micro_batch])
        last_x = _decompress_topk_nbits(
            last_compressed_activ, last_compressed_masks, last_compressed_scale, 
            bits=self.bits_act, k=self.k_act, original_shape=self.activ_shape,)
        # compresss delta
        delta = x - last_x
        compressed_delta = compress_flexible_nbits(
            delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        # update cache
        delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
        x = last_x + delta
        compressed_x = _compress_topk_nbits(
            x, bits=self.bits_act, k=self.k_act, scale_method='max')
        a_cp = tensor_to_cupy(compressed_x[0])
        m_cp = tensor_to_cupy(compressed_x[1])
        s_cp = tensor_to_cupy(compressed_x[2])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        m_cp.get(out=self.np_com_masks_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_activ_buffers[i_micro_batch].set(self.np_dec_activ_buffers[i_micro_batch])
        self.cp_dec_masks_buffers[i_micro_batch].set(self.np_dec_masks_buffers[i_micro_batch])
        self.cp_dec_scale_buffers[i_micro_batch].set(self.np_dec_scale_buffers[i_micro_batch])
        last_compressed_activ = cupy_to_tensor(self.cp_dec_activ_buffers[i_micro_batch])
        last_compressed_masks = cupy_to_tensor(self.cp_dec_masks_buffers[i_micro_batch])
        last_compressed_scale = cupy_to_tensor(self.cp_dec_scale_buffers[i_micro_batch])
        last_x = _decompress_topk_nbits(
            last_compressed_activ, last_compressed_masks, last_compressed_scale, 
            bits=self.bits_act, k=self.k_act, original_shape=self.activ_shape,)
        # decompress delta
        delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
        # update cache
        x = last_x + delta
        compressed_x = _compress_topk_nbits(
            x, bits=self.bits_act, k=self.k_act, scale_method='max')
        a_cp = tensor_to_cupy(compressed_x[0])
        m_cp = tensor_to_cupy(compressed_x[1])
        s_cp = tensor_to_cupy(compressed_x[2])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        m_cp.get(out=self.np_dec_masks_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x
    
    
    def no_compress(self, x, i_micro_batch):
        # update cache
        compressed_x = _compress_topk_nbits(
            x, bits=self.bits_act, k=self.k_act, scale_method='max')
        a_cp = tensor_to_cupy(compressed_x[0])
        m_cp = tensor_to_cupy(compressed_x[1])
        s_cp = tensor_to_cupy(compressed_x[2])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        m_cp.get(out=self.np_com_masks_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return x
        
    def no_decompress(self, x, i_micro_batch):
        # update cache
        compressed_x = _compress_topk_nbits(
            x, bits=self.bits_act, k=self.k_act, scale_method='max')
        a_cp = tensor_to_cupy(compressed_x[0])
        m_cp = tensor_to_cupy(compressed_x[1])
        s_cp = tensor_to_cupy(compressed_x[2])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        m_cp.get(out=self.np_dec_masks_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x
    
    
    
def _compress_topk_nbits_batch(x, k, bits, scale_method='max'):
    batch_size = x.size(0)
    
    # 1. sparsify x (keep batch dims)
    x_flat = x.view(-1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.bool)
    masks[indexes] = 1
    masks = masks.view(x.shape)
    x = x * masks
    
    # 2. quantize x        
    uint_x, scales = _compress_nbits(x, bits, scale_method=scale_method, scale_dims=(0,1,))
    topk_uint_x = uint_x[masks].view(k)
    if bits == 8:
        pass
    elif bits == 4:
        x0, x1 = topk_uint_x.chunk(2, -1)
        topk_uint_x = (x0 << 4) + x1
    elif bits == 2:
        x0, x1, x2, x3 = topk_uint_x.chunk(4, -1)
        topk_uint_x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3
    else:
        raise Exception('not support bits')
    masks = cupy_to_tensor(
        cupy.packbits(tensor_to_cupy(masks))
    ).view(-1)
    return topk_uint_x, masks, scales
    
def _decompress_topk_nbits_batch(topk_uint_x, masks, scales, k, bits, original_shape):
    masks = cupy_to_tensor(
        cupy.unpackbits(tensor_to_cupy(masks))
    )
    masks = masks.view(original_shape)
    if bits == 8:
        pass
    elif bits == 4:
        bitmask = 15
        x0 = (topk_uint_x >> 4)
        x1 = (topk_uint_x & bitmask)
        topk_uint_x = torch.cat([x0, x1], -1)
    elif bits == 2:
        bitmask = 3
        x0 = (topk_uint_x >> 6)
        x1 = (topk_uint_x >> 4) & bitmask
        x2 = (topk_uint_x >> 2) & bitmask
        x3 = topk_uint_x & bitmask
        topk_uint_x = torch.cat([x0, x1, x2, x3], -1)
    else:
        raise Exception('not support bits')
    uint_x = torch.zeros(original_shape, dtype=torch.uint8, device=topk_uint_x.device) + (1<<bits-1)
    uint_x[masks] = topk_uint_x.view(-1)
    x = _decompress_nbits(uint_x, scales, bits)
    return x
    
class TopKDeltaCompressor(DeltaCompressor):
    def __init__(
        self, bits=4, ratio=0.1,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in [1, 8]
        ratio in [0, 1]
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.ratio = ratio
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        assert scale_dims == (0,1)
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future_read = None
        self.future_write = None
        
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        self.k_delta = int(micro_batch_size * seq_length * embedding_dim * self.ratio)
        self.k_delta = self.k_delta + ((8//self.bits) - self.k_delta % (8//self.bits))
        self.k_delta = max(self.k_delta, 1)
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.scale_shape = scale_shape
        
        # Activation Cache
        self.tmp_f = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache = np.memmap(
            self.tmp_f, mode='w+', dtype=np.float16, shape=(MAX_CACHE_SIZE, 2, seq_length, embedding_dim),
        )
        
        # Communication Buffers
        self.compressed_delta_shape = (self.k_delta*self.bits//8,)
        self.compressed_masks_shape = (micro_batch_size, seq_length*embedding_dim//8,)
        self.compressed_scale_shape = (1,1,embedding_dim,)
        self.buffers = [
            (
                torch.zeros(self.compressed_delta_shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(self.compressed_masks_shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(self.compressed_scale_shape, requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # CPU RAM Buffers
        self.np_dec_buffers = [
            pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_buffers = [
            pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        
        # GPU RAM Buffers
        self.cp_dec_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x, i_micro_batch):
        # get cache
        self.cp_com_buffers[i_micro_batch].set(self.np_com_buffers[i_micro_batch])
        last_x = cupy_to_tensor(self.cp_com_buffers[i_micro_batch])
        delta = x - last_x
        # compresss delta
        compressed_delta = _compress_topk_nbits_batch(
            delta, bits=self.bits, k=self.k_delta, scale_method=self.scale_method)
        # update cache
        delta = _decompress_topk_nbits_batch(
            *compressed_delta, bits=self.bits, k=self.k_delta, original_shape=self.activ_shape)
        x = last_x + delta
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_buffers[i_micro_batch].set(self.np_dec_buffers[i_micro_batch])
        last_x = cupy_to_tensor(self.cp_dec_buffers[i_micro_batch])
        # decompress delta
        delta = _decompress_topk_nbits_batch(
            *delta, bits=self.bits, k=self.k_delta, original_shape=self.activ_shape)
        # update cache
        x = last_x + delta
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_dec_buffers[i_micro_batch])
        return x
        
