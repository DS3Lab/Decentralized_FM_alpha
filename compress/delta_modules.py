
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy
from torch.utils.dlpack import to_dlpack, from_dlpack
import concurrent.futures
import tempfile

from .fixpoint import *
from . import flag

def _cupy_to_tensor(x):
    return from_dlpack(x.toDlpack())

def _tensor_to_cupy(x):
    return cupy.fromDlpack(to_dlpack(x))


def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


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
            self.tmp_f, mode='w+', dtype=np.float16, shape=(100000, 2, seq_length, embedding_dim),
        )
        
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
            _pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_buffers = [
            _pin_memory(np.zeros(self.activ_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        
        # GPU RAM Buffers
        self.cp_dec_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        self.cp_com_buffers = [
            cupy.empty(self.activ_shape, dtype=np.float16) for _ in range(batch_size//micro_batch_size)
        ]
        
    def _read_from_cache(self, sample_ids):
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
        self.future_read = self.executor.submit(self._write_to_cache, sample_ids)
        
    def compress(self, x, i_micro_batch):
        # get cache
        self.cp_com_buffers[i_micro_batch].set(self.np_com_buffers[i_micro_batch])
        last_x = _cupy_to_tensor(self.cp_com_buffers[i_micro_batch])
        delta = x - last_x
        # compresss delta
        compressed_delta = compress_flexible_nbits(delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        # update cache
        delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
        x = last_x + delta
        x_cp = _tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_buffers[i_micro_batch].set(self.np_dec_buffers[i_micro_batch])
        last_x = _cupy_to_tensor(self.cp_dec_buffers[i_micro_batch])
        # decompress delta
        delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
        # update cache
        x = last_x + delta
        x_cp = _tensor_to_cupy(x.half())
        x_cp.get(out=self.np_dec_buffers[i_micro_batch])
        return x
    
    def no_compress(self, x, i_micro_batch):
        # update cache
        x_cp = _tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return x
        
    def no_decompress(self, x, i_micro_batch):
        # update cache
        x_cp = _tensor_to_cupy(x.half())
        x_cp.get(out=self.np_dec_buffers[i_micro_batch])
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        self._wait_read()
        self._wait_write()
        
        if not flag.FLAG_DISABLE_COMPRESSION:
#             if dst == 1 and i_micro_batch == 0:
#                 print('0 send', x.view(-1)[:10])
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
#             if src == 0 and i_micro_batch == 0:
#                 print('1 get:', x.view(-1)[:10])
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
                100000, 2, *self.compressed_activ_shape,
            ),
        )
        self.tmp_f2 = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache_scale = np.memmap(
            self.tmp_f2, mode='w+', dtype=np.float16, shape=(
                100000, 2, *self.compressed_scale_shape,
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
            _pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_dec_scale_buffers = [
            _pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_activ_buffers = [
            _pin_memory(np.zeros(self.micro_compressed_activ_shape, dtype=np.uint8)) for _ in range(batch_size//micro_batch_size)
        ]
        self.np_com_scale_buffers = [
            _pin_memory(np.zeros(self.micro_compressed_scale_shape, dtype=np.float16)) for _ in range(batch_size//micro_batch_size)
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
        last_compressed_activ = _cupy_to_tensor(self.cp_com_activ_buffers[i_micro_batch])
        last_compressed_scale = _cupy_to_tensor(self.cp_com_scale_buffers[i_micro_batch])
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
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = _tensor_to_cupy(compressed_x[0])
        s_cp = _tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return compressed_delta
        
    def decompress(self, delta, i_micro_batch):
        # get cache
        self.cp_dec_activ_buffers[i_micro_batch].set(self.np_dec_activ_buffers[i_micro_batch])
        self.cp_dec_scale_buffers[i_micro_batch].set(self.np_dec_scale_buffers[i_micro_batch])
        last_compressed_activ = _cupy_to_tensor(self.cp_dec_activ_buffers[i_micro_batch])
        last_compressed_scale = _cupy_to_tensor(self.cp_dec_scale_buffers[i_micro_batch])
        last_x = decompress_nbits(
            last_compressed_activ, last_compressed_scale, 
            bits=self.bits_act)
        # decompress delta
        delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
        # update cache
        x = last_x + delta
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = _tensor_to_cupy(compressed_x[0])
        s_cp = _tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x
    
    
    def no_compress(self, x, i_micro_batch):
        # update cache
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = _tensor_to_cupy(compressed_x[0])
        s_cp = _tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_com_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_com_scale_buffers[i_micro_batch])
        return x
        
    def no_decompress(self, x, i_micro_batch):
        # update cache
        compressed_x = compress_nbits(
            x, self.bits_act, scale_method='max', scale_dims=(1,))
        a_cp = _tensor_to_cupy(compressed_x[0])
        s_cp = _tensor_to_cupy(compressed_x[1])
        a_cp.get(out=self.np_dec_activ_buffers[i_micro_batch])
        s_cp.get(out=self.np_dec_scale_buffers[i_micro_batch])
        return x