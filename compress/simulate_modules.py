
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

import faiss
import faiss.contrib.torch_utils


class SimulateTestCompression:
    def __init__(self, bits=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        self.bits = bits
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=torch.float32):
        self.buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        self.embedding_dim = embedding_dim
        self.tmp_f = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.cache = np.memmap(
            self.tmp_f, mode='w+', dtype=np.float16, shape=(MAX_CACHE_SIZE, seq_length, embedding_dim),
        )
        self.tmp_f2 = tempfile.NamedTemporaryFile(dir='/tmp/')
        self.simulate_cache = np.memmap(
            self.tmp_f2, mode='w+', dtype=np.float16, shape=(MAX_CACHE_SIZE, seq_length, embedding_dim),
        )
        self.ccache = torch.randn([int(1e4), embedding_dim], dtype=dtype, device=device).half() / 100.
        self.res = faiss.StandardGpuResources()
        self.count = 0
        
    def compress(self, x):
        with torch.no_grad():
            
            if self.count < self.ccache.size(0):
                self.count += x.size(0) * x.size(1)
            
            if self.count < self.ccache.size(0):
                b, e = self.count - x.size(0) * x.size(1), self.count
                cache_idx = torch.arange(b, e).view(x.shape[:-1])
            else:
                x_ = x.view(-1, x.size(-1)).half()
                cache_idx = faiss.knn_gpu(self.res, x_, self.ccache, 1, metric=faiss.METRIC_L1)[1]
                cache_idx = cache_idx.view(x.shape[:-1])
            
            if flag.FLAG_DISABLE_COMPRESSION:
                
                x_to_save = x.clone()
#                 self.ccache.data *= 0.999
                self.ccache.data[cache_idx] = self.ccache.data[cache_idx] * 0.9 + 0.1 * x_to_save.half()
                return x
        
            x_hat = self.ccache[cache_idx]
            delta = x - x_hat
            compressed_delta = compress_nbits(
                delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
            decompressed_delta = decompress_nbits(*compressed_delta, self.bits)
            ### send & recv (delta, cache_idx)
            new_x = x_hat + decompressed_delta
            x_to_save = new_x.clone()
#             self.ccache.data *= 0.999
            self.ccache.data[cache_idx] = self.ccache.data[cache_idx] * 0.9 + 0.1 * x_to_save.half()
        
        return new_x
        
    def decompress(self, x):
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        with stream:
            x = self.compress(x)
        comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        comm.recv(recv_buffer, src=src, stream=stream)
        return recv_buffer
    

class SimulateTestCompressionLR:
    def __init__(self, bits=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        self.bits = bits
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=torch.float32):
        self.buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
        self.embedding_dim = embedding_dim
        self.cache = torch.randn([int(1e5), embedding_dim], dtype=dtype, device=device) / 100.
        self.count = 0
        
    def compress(self, x):
        
        print('low rank compress')
        
        if flag.FLAG_DISABLE_COMPRESSION:
            return x
        
        with torch.no_grad():
            A = x.view(-1, x.size(-1))
            q = int(A.numel() / (A.size(0) + A.size(1)) / 4)
            U, S, Vh = torch.svd_lowrank(A, q=q, niter=10)
            A = U @ torch.diag(S) @ Vh.T
            new_x = A.view(x.shape)
        
        return new_x
        
    def decompress(self, x):
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        with stream:
            x = self.compress(x)
        comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        comm.recv(recv_buffer, src=src, stream=stream)
        return recv_buffer
    

    
    
    
MAX_CACHE_SIZE = 12000


class SimulateDeltaCompression:
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
        self.simulate_cache = np.zeros(
            (MAX_CACHE_SIZE, seq_length, embedding_dim), dtype=np.float32,
        )
        self.simulate_diff = np.zeros(
            (MAX_CACHE_SIZE, seq_length, embedding_dim), dtype=np.float32,
        )
        self.sample_ids = None
        
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
        self.sample_ids = sample_ids
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
        
    def simulate_kmeans(self):
#         pass
        if hasattr(self, 'kmeans'):
            init_centroids = self.kmeans.centroids
        else:
            init_centroids = None
        self.kmeans = faiss.Kmeans(d=self.cache.shape[-1], k=int(1e5), niter=1, nredo=1)
        data_train = self.cache[:, 1].reshape(-1, self.cache.shape[-1]).astype('float32')
        data_train /= (np.linalg.norm(data_train, axis=1, keepdims=True) + 1e-8)
        self.kmeans.train(
            data_train, init_centroids=init_centroids,
        )
        
    def simulate_cluster_compress(self, x, i_micro_batch):
        sample_ids = self.sample_ids[i_micro_batch*self.micro_batch_size:(i_micro_batch+1)*self.micro_batch_size]
        self.simulate_cache[sample_ids] = (x).detach().cpu().numpy()
        
#         kmeans = self.kmeans
        
#         sample_ids = self.sample_ids[i_micro_batch*self.micro_batch_size:(i_micro_batch+1)*self.micro_batch_size]
#         x_ = x.view(-1, x.size(-1)).detach().cpu().numpy()
#         x_ /= (np.linalg.norm(x_, axis=1, keepdims=True) + 1e-8)
#         x_hat = kmeans.centroids[kmeans.index.search(x_, 1)[1].squeeze(-1)]
#         x_hat = torch.from_numpy(x_hat).view(x.shape).to(x.device)
#         x_hat *= x.norm(dim=-1, keepdim=True)
#         delta = x - x_hat
#         self.simulate_diff[sample_ids] = (delta).detach().cpu().numpy()
#         compressed_delta = compress_flexible_nbits(delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
#         delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
#         self.simulate_cache[sample_ids] = (x_hat + delta).detach().cpu().numpy()
        
    def compress(self, x, i_micro_batch):
        # get cache
#         self.simulate_cluster_compress(x, i_micro_batch)
        self.cp_com_buffers[i_micro_batch].set(self.np_com_buffers[i_micro_batch])
        last_x = cupy_to_tensor(self.cp_com_buffers[i_micro_batch])
        delta = x - last_x
        # compresss delta
        #####
#         compressed_delta = compress_flexible_nbits(delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
#         delta = decompress_flexible_nbits(*compressed_delta, self.bits, self.activ_shape)
        A = delta.view(-1, delta.size(-1))
        q = int(A.numel() / (A.size(0) + A.size(1)) / 4)
        U, S, Vh = torch.svd_lowrank(A, q=q, niter=10)
        U = U.half().float()
        S = S.half().float()
        Vh = Vh.half().float()
        A = U @ torch.diag(S) @ Vh.T
        delta = A.view(delta.shape)
        #####
        x = last_x + delta
        x_cp = tensor_to_cupy(x.half())
        x_cp.get(out=self.np_com_buffers[i_micro_batch])
        return x
        
    def decompress(self, x, i_micro_batch):
        # get cache
#         self.cp_dec_buffers[i_micro_batch].set(self.np_dec_buffers[i_micro_batch])
#         last_x = cupy_to_tensor(self.cp_dec_buffers[i_micro_batch])
#         # decompress delta
#         delta = decompress_flexible_nbits(*delta, self.bits, self.activ_shape)
#         # update cache
#         x = last_x + delta
#         x_cp = tensor_to_cupy(x.half())
#         x_cp.get(out=self.np_dec_buffers[i_micro_batch])
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
            comm.send(_data, dst=dst, stream=stream)
        else:
            with stream:
                x = self.no_compress(x, i_micro_batch=i_micro_batch)
            comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        self._wait_read()
        self._wait_write()
        if not flag.FLAG_DISABLE_COMPRESSION:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            with stream:
                x = self.decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            with stream:
                x = self.no_decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        
        
        
        
class SimulateDeltaCompression:
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
        self.cache = self.simulate_cache = np.zeros(
            (MAX_CACHE_SIZE, seq_length, embedding_dim), dtype=np.float32,
        )
        # Info: ensure it has content, so the profiling will be accurate
        #self.cache.fill(0)
        self.simulate_cache = np.zeros(
            (MAX_CACHE_SIZE, seq_length, embedding_dim), dtype=np.float32,
        )
        self.simulate_diff = np.zeros(
            (MAX_CACHE_SIZE, seq_length, embedding_dim), dtype=np.float32,
        )
        self.sample_ids = None
        
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
        
        self.ccache = torch.randn([int(1e4), embedding_dim], dtype=dtype, device=device).half() / 100.
        self.res = faiss.StandardGpuResources()
        self.count = 0
        self.counts = np.zeros(int(1e4))
        
        
    def _read_from_cache(self, sample_ids):
#         time.sleep(1)
#         activations = self.cache[sample_ids]
#         a_dec, a_com = activations[:, 0], activations[:, 1]
        self.sample_ids = sample_ids
#         for i in range(self.batch_size//self.micro_batch_size):
#             self.np_dec_buffers[i][:] = a_dec[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
#             self.np_com_buffers[i][:] = a_com[i*self.micro_batch_size:(i+1)*self.micro_batch_size]
            
    def _write_to_cache(self, sample_ids):
#         dec_activations = np.concatenate(self.np_dec_buffers, 0)
#         com_activations = np.concatenate(self.np_com_buffers, 0)
#         activations = np.stack([dec_activations, com_activations], 1)
#         self.cache[sample_ids] = activations
#         self.cache.flush()
        pass
        
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
        
        sample_ids = self.sample_ids[i_micro_batch*self.micro_batch_size:(i_micro_batch+1)*self.micro_batch_size]
        
        self.cache[sample_ids] = x.detach().cpu().numpy()
        
        with torch.no_grad():
            
            if self.count < self.ccache.size(0):
                self.count += x.size(0) * x.size(1)
            
            if self.count < self.ccache.size(0):
                b, e = self.count - x.size(0) * x.size(1), self.count
                cache_idx = torch.arange(b, e).view(x.shape[:-1])
            else:
                x_ = x.view(-1, x.size(-1)).half()
                cache_idx = faiss.knn_gpu(self.res, x_, self.ccache, 1, metric=faiss.METRIC_L1)[1]
                cache_idx = cache_idx.view(x.shape[:-1])
            
            self.counts[cache_idx.cpu()] += 1
            
            if flag.FLAG_DISABLE_COMPRESSION:
#                 self.ccache.data *= 0.999
                self.ccache.data[cache_idx] = self.ccache.data[cache_idx] * 0.9 + 0.1 * x.half()
                return x
        
            x_hat = self.ccache[cache_idx]
            delta = x - x_hat
            
            self.simulate_diff[sample_ids] = delta.detach().cpu().numpy()
            
            compressed_delta = compress_nbits(
                delta, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
            decompressed_delta = decompress_nbits(*compressed_delta, self.bits)
            ### send & recv (delta, cache_idx)
            new_x = x_hat + decompressed_delta
            x_to_save = new_x.clone()
#             self.ccache.data *= 0.999
            self.ccache.data[cache_idx] = self.ccache.data[cache_idx] * 0.9 + 0.1 * x_to_save.half()
        
        self.simulate_cache[sample_ids] = new_x.detach().cpu().numpy()
        
        return new_x
        
    def decompress(self, x, i_micro_batch):
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
            comm.send(_data, dst=dst, stream=stream)
        else:
            with stream:
                x = self.no_compress(x, i_micro_batch=i_micro_batch)
            comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        self._wait_read()
        self._wait_write()
        if not flag.FLAG_DISABLE_COMPRESSION:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            with stream:
                x = self.decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            with stream:
                x = self.no_decompress(recv_buffer, i_micro_batch=i_micro_batch)
            return x
        
