import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import math
from comm.comm_utils import *
from .flatten_utils import flatten_params, flatten_tensors
from compress.fixpoint import *
from compress.sparsification import *
from compress import flag
import cupy

import os

if 'SYNC_STEPS' not in os.environ:
    sync_steps = 25
    sync_prob = 1.0 / sync_steps
    global_sync_steps = 1
else:
    sync_steps = int(os.environ['SYNC_STEPS'])
    sync_prob = 1.0 / sync_steps
    global_sync_steps = 1
    
global_lr = float(os.environ.get('GLOBAL_LR', 1.0))
quantization_bits = int(os.environ.get('QUANT_BITS', 8))
quantization_bucket_size = int(os.environ.get('QUANT_BUCKET_SIZE', 128))
top_k_ratio = float(os.environ.get('TOPK_RATIO', 0.5))

import zlib

try:
    import lz4.frame
except:
    pass

def _lossless_compress(data):
    assert data.dtype == torch.uint8
    raw = data.detach().cpu().numpy().tobytes()
    # enc = lz4.frame.compress(raw)
    enc = zlib.compress(raw)
    rate = len(raw) / len(enc)
    # print(f'c-rate: {rate}x')
    # if rate < 1.5:
    #     print(raw[:1000])
        # assert False
    return enc

def _lossless_decompress(enc):
    # dec = lz4.frame.decompress(enc)
    dec = zlib.decompress(enc)
    data = torch.frombuffer(dec, dtype=torch.uint8)
    return data

class SlotSGDDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=False):
        # assert not flatten
        self.dp_bits = args.dp_bits
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.pp_comm = get_pipeline_parallel_comm()
        self.pp_rank = get_pipeline_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        
        if self.flatten:
            _params = []
            for i_group, group in enumerate(self.optimizer.optimizer.param_groups):
                for i_para, para in enumerate(group["params"]):
                    _params.append(para)
            self.flatten_para = flatten_tensors(_params)
            print("Flattened parameter number: {}, element size: {}."
                  .format(self.flatten_para.data.numel(), self.flatten_para.data.element_size()))
            
            
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))
        
        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            # assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.gather_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.gather_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.worker_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.worker_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
        self.dp_state_dict = {}

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)
            
    def allreduce_parameters(self):
        self._local_parameters_backup = [
            p.data.clone() for p in self.module.parameters()
        ]
        torch.cuda.synchronize()
        self.dp_comm.barrier()
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            # self.dp_comm_stream.wait_event(self.backward_ready_event)
            for name, para in self.module.named_parameters():
                # self.profile_mark_allreduce_start(name)
                para.data /= self.dp_group_size
                self.dp_comm.all_reduce(para.data, stream=cupy_dp_stream)
                # self.profile_mark_allreduce_end(name)
            # self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)
        torch.cuda.synchronize()
        self.dp_comm.barrier()

    def rollback_parameters(self):
        if not hasattr(self, '_local_parameters_backup'):
            return
        
        for p, p_local in zip(self.module.parameters(), self._local_parameters_backup):
            p.data[:] = p_local.data
            
        del self._local_parameters_backup
            
    def _allreduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            if self.flatten:
                # self.profile_mark_allreduce_start()
                self.dp_comm.all_reduce(self.flatten_para.grad, stream=cupy_dp_stream)
                self.profile_mark_allreduce_end()
            else:
                for name, para in self.module.named_parameters():
                    if para.grad is None:
                        continue
                    # self.profile_mark_allreduce_start(name)
                    self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
                    # self.profile_mark_allreduce_end(name)
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
    def _compress(self, x):
        # return x
        dtype = x.dtype
        shape = x.shape
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with cupy_dp_stream:
                
                k = max(int(top_k_ratio * x.numel()), 1)
                if k >= quantization_bucket_size:
                    # ensure dividable
                    k = k // quantization_bucket_size * quantization_bucket_size
                else:
                    # bucket_size will be set to k internally
                    pass
                    
                values, masks, indices = compress_topk(x, k, return_indices=True)

                values_q, scales_q = compress_flexible_nbits_by_bucket(values, bits=quantization_bits, scale_method='max', bucket_size=quantization_bucket_size)
                
                return (values_q, scales_q, masks), (dtype, shape, values.shape)
    
    def _decompress(self, x_hat, meta_data):
        
        values_q, scales_q, masks = x_hat
        x_dtype, x_shape, values_shape = meta_data
        
        values = decompress_flexible_nbits_by_bucket(values_q, scales_q, bits=quantization_bits, original_shape=values_shape, bucket_size=quantization_bucket_size)
                    
        x = decompress_topk(values, masks, x_shape)
        x = x.to(x_dtype)
        
        return x
            
    def _partial_sync(self):
        
        if self.flatten:
            
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with torch.cuda.stream(self.dp_comm_stream), cupy_dp_stream:
                
                self.dp_comm_stream.record_event(self.sync_gradients_start_event)
                
                name = 'model'
                para = self.flatten_para
                
                dp_state_dict = self.dp_state_dict

                if name not in dp_state_dict:

                    # comm mask
                    comm_mask_list = []
                    comm_data_list = []
                    for i in range(self.dp_group_size):
                        para_shape = list(para.shape)
                        assert para_shape[0] == para_shape[0] // self.dp_group_size * self.dp_group_size
                        para_shape[0] = para_shape[0] // self.dp_group_size
                        comm_mask = torch.zeros(para_shape, dtype=torch.bool, device=para.device)
                        comm_mask.view(-1)[::sync_steps] = True
                        n_potisive = comm_mask.sum().item() // quantization_bucket_size * quantization_bucket_size
                        if n_potisive != 0:
                            comm_mask.view(-1)[comm_mask.view(-1).cumsum(-1) > n_potisive] = False
                            assert comm_mask.sum().item() == n_potisive
                        else:
                            comm_mask[:] = True
                        print('comm_mask:', comm_mask.sum().item(), comm_mask.shape)
                        comm_mask_list.append(comm_mask)

                    # global para
                    global_para = para.data.half()

                    # server error
                    # server_error = torch.zeros_like(global_para.chunk(self.dp_group_size, 0)[self.dp_rank])
                    server_error = torch.zeros(
                        comm_mask_list[self.dp_rank].shape, dtype=torch.float16, device=para.device,
                    )

                    # print('server error shape:', server_error.shape)
                    dp_state_dict[name] = {
                        "comm_mask_list": comm_mask_list,
                        # "comm_data_list": comm_data_list,
                        "global_para": global_para,
                        "server_error": server_error,
                    }
                else:
                    for i in range(self.dp_group_size):
                        dp_state_dict[name]['comm_mask_list'][i] = dp_state_dict[name]['comm_mask_list'][i].roll(1)

                comm_mask_list = dp_state_dict[name]["comm_mask_list"]
                comm_data_list = comm_data_list = [None for _ in comm_mask_list]
                global_para = dp_state_dict[name]["global_para"]
                chunk_size = global_para.size(0) // self.dp_group_size
                server_error = dp_state_dict[name]["server_error"]
                server_mask = comm_mask_list[self.dp_rank]

                for i in range(self.dp_group_size):
                    comm_mask = comm_mask_list[i]
                    # comm_data_list[i].data[:] = (para[i*chunk_size:(i+1)*chunk_size][comm_mask] - global_para[i*chunk_size:(i+1)*chunk_size][comm_mask]).half()
                    comm_data_list[i] = (para[i*chunk_size:(i+1)*chunk_size][comm_mask] - global_para[i*chunk_size:(i+1)*chunk_size][comm_mask]).half()
                    
                comm_data_compressed_list = []
                comm_data_meta_list = []
                for x in comm_data_list:
                    data, meta_data = self._compress(x)
                    # z = self._decompress(data, meta_data)
                    # assert z.shape == x.shape
                    comm_data_compressed_list.append(data)
                    comm_data_meta_list.append(meta_data)
                    del x
                del comm_data_list
                comm_buffer_list = [[torch.zeros_like(x) for x in x_tuple] for x_tuple in comm_data_compressed_list]
                
                # revert
                for i in range(self.dp_group_size):
                    # print('A', len(comm_data_compressed_list[i]))
                    _data_compressed = self._decompress(comm_data_compressed_list[i], comm_data_meta_list[i])
                    para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] -= _data_compressed
                    del _data_compressed

                # print(f'do first group r{self.global_rank} - {i_group}/{len(self.optimizer.optimizer.param_groups)} - {i_para}/{len(group["params"])}  - {para.shape}')
                # self.dp_comm.barrier()
                    
                cupy.cuda.nccl.groupStart()
                for i in range(self.dp_group_size):
                    for j, to_send in enumerate(comm_data_compressed_list[i]):
                        self.dp_comm.send(
                            to_send, dst=i, stream=cupy_dp_stream)
                        # if j==0:
                        #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                        # if to_send.dtype == torch.uint8 and to_send.numel() > 1000:
                        #     _enc = _lossless_compress(to_send)
                        #     send_bits += len(_enc) * 8
                        # else:
                        #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                    for to_recv in comm_buffer_list[i]:
                        self.dp_comm.recv(
                            to_recv, src=i, stream=cupy_dp_stream)
                        # recv_bits += to_recv.numel() * torch.finfo(to_recv.dtype).bits if to_recv.is_floating_point() else to_recv.numel() * torch.iinfo(to_recv.dtype).bits
                cupy.cuda.nccl.groupEnd()

                server_data = self._decompress([z for z in comm_buffer_list[0]], comm_data_meta_list[0]) / len(comm_buffer_list)
                for i in range(1, self.dp_group_size):
                    server_data.data += self._decompress([z for z in comm_buffer_list[i]], comm_data_meta_list[i]) / len(comm_buffer_list)
                server_data.add_(server_error[server_mask])
                server_data_compressed, server_data_meta = self._compress(server_data)
                server_error.data[server_mask] = (server_data - self._decompress(server_data_compressed, server_data_meta))
                
                # print(f'do second group r{self.global_rank} - {i_group}/{len(self.optimizer.optimizer.param_groups)} - {i_para}/{len(group["params"])} - {para.shape}')
                # self.dp_comm.barrier()

                cupy.cuda.nccl.groupStart()
                for i in range(self.dp_group_size):
                    for j, to_send in enumerate(server_data_compressed):
                        self.dp_comm.send(
                            to_send, dst=i, stream=cupy_dp_stream)
                        # if j==0:
                        #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                        # if to_send.dtype == torch.uint8  and to_send.numel() > 1000:
                        #     _enc = _lossless_compress(to_send)
                        #     send_bits += len(_enc) * 8
                        # else:
                        #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                    for to_recv in comm_buffer_list[i]:
                        self.dp_comm.recv(
                            to_recv, src=i, stream=cupy_dp_stream)
                        # recv_bits += to_recv.numel() * torch.finfo(to_recv.dtype).bits if to_recv.is_floating_point() else to_recv.numel() * torch.iinfo(to_recv.dtype).bits
                cupy.cuda.nccl.groupEnd()

                for i in range(self.dp_group_size):
                    
                    _data = self._decompress([z for z in comm_buffer_list[i]], comm_data_meta_list[i])
                    para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                    global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data
                    
                    del _data
                    
                    # para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] = \
                    # global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]].float()

                self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
        else:
            
            original_bits = 0
            send_bits = 0
            recv_bits = 0
            
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            with torch.cuda.stream(self.dp_comm_stream), cupy_dp_stream:
                
                # for name, para in self.module.named_parameters():
                for i_group, group in enumerate(self.optimizer.optimizer.param_groups):
                    for i_para, para in enumerate(group["params"]):
                        
                        # original_bits += para.numel() * 16 #torch.finfo(para.dtype).bits
                        
                        para = para.view(-1)
                        
                        name = f"{i_group}-{i_para}"
                    
                        dp_state_dict = self.dp_state_dict

                        if name not in dp_state_dict:

                            # comm mask
                            comm_mask_list = []
                            comm_data_list = []
                            for i in range(self.dp_group_size):
                                para_shape = list(para.shape)
                                assert para_shape[0] == para_shape[0] // self.dp_group_size * self.dp_group_size
                                para_shape[0] = para_shape[0] // self.dp_group_size
                                comm_mask = torch.zeros(para_shape, dtype=torch.bool, device=para.device)
                                comm_mask.view(-1)[::sync_steps] = True
                                n_potisive = comm_mask.sum().item() // quantization_bucket_size * quantization_bucket_size
                                if n_potisive != 0:
                                    comm_mask.view(-1)[comm_mask.view(-1).cumsum(-1) > n_potisive] = False
                                    assert comm_mask.sum().item() == n_potisive
                                else:
                                    comm_mask[:] = True
                                print('comm_mask:', comm_mask.sum().item(), comm_mask.shape)
                                comm_mask_list.append(comm_mask)

                                # data to send
                                comm_data = torch.zeros(
                                    comm_mask.sum().item(), dtype=torch.float16, device=para.device
                                )
                                # comp_data, _ = self._compress(comm_data)
                                # print('comm_data:', [cd.numel() for cd in comp_data])
                                comm_data_list.append(comm_data)

                            # global para
                            global_para = para.data.half()

                            # server error
                            # server_error = torch.zeros_like(global_para.chunk(self.dp_group_size, 0)[self.dp_rank])
                            server_error = torch.zeros(
                                comm_mask_list[self.dp_rank].shape, dtype=torch.float16, device=global_para.device,
                            )

                            print('server error shape:', server_error.shape)
                            dp_state_dict[name] = {
                                "comm_mask_list": comm_mask_list,
                                "comm_data_list": comm_data_list,
                                "global_para": global_para,
                                "server_error": server_error,
                            }
                        else:
                            for i in range(self.dp_group_size):
                                dp_state_dict[name]['comm_mask_list'][i] = dp_state_dict[name]['comm_mask_list'][i].roll(1)

                        comm_mask_list = dp_state_dict[name]["comm_mask_list"]
                        comm_data_list = dp_state_dict[name]["comm_data_list"]
                        global_para = dp_state_dict[name]["global_para"]
                        chunk_size = global_para.size(0) // self.dp_group_size
                        server_error = dp_state_dict[name]["server_error"]
                        server_mask = comm_mask_list[self.dp_rank]

                        for i in range(self.dp_group_size):
                            comm_mask = comm_mask_list[i]
                            comm_data_list[i].data[:] = (para[i*chunk_size:(i+1)*chunk_size][comm_mask] - global_para[i*chunk_size:(i+1)*chunk_size][comm_mask]).half()
                            
                        comm_data_compressed_list = []
                        comm_data_meta_list = []
                        for x in comm_data_list:
                            data, meta_data = self._compress(x)
                            z = self._decompress(data, meta_data)
                            assert z.shape == x.shape
                            comm_data_compressed_list.append(data)
                            comm_data_meta_list.append(meta_data)
                        comm_buffer_list = [[torch.zeros_like(x) for x in x_tuple] for x_tuple in comm_data_compressed_list]
                        
                        # revert
                        for i in range(self.dp_group_size):
                            # print('A', len(comm_data_compressed_list[i]))
                            _data_compressed = self._decompress(comm_data_compressed_list[i], comm_data_meta_list[i])
                            para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] -= _data_compressed

                        cupy.cuda.nccl.groupStart()
                        for i in range(self.dp_group_size):
                            for j, to_send in enumerate(comm_data_compressed_list[i]):
                                self.dp_comm.send(
                                    to_send, dst=i, stream=cupy_dp_stream)
                                # if j==0:
                                #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                                # if to_send.dtype == torch.uint8 and to_send.numel() > 1000:
                                #     _enc = _lossless_compress(to_send)
                                #     send_bits += len(_enc) * 8
                                # else:
                                #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                            for to_recv in comm_buffer_list[i]:
                                self.dp_comm.recv(
                                    to_recv, src=i, stream=cupy_dp_stream)
                                # recv_bits += to_recv.numel() * torch.finfo(to_recv.dtype).bits if to_recv.is_floating_point() else to_recv.numel() * torch.iinfo(to_recv.dtype).bits
                        cupy.cuda.nccl.groupEnd()

                        # print('B', len(comm_buffer_list[0]))
                        server_data = self._decompress(comm_buffer_list[0], comm_data_meta_list[0]) / len(comm_buffer_list)
                        for i in range(1, self.dp_group_size):
                            # print('B', len(comm_buffer_list[i]))
                            server_data.data += self._decompress(comm_buffer_list[i], comm_data_meta_list[i]) / len(comm_buffer_list)
                        server_data.add_(server_error[server_mask])
                        server_data_compressed, server_data_meta = self._compress(server_data)
                        server_error.data[server_mask] = (server_data - self._decompress(server_data_compressed, server_data_meta))

                        cupy.cuda.nccl.groupStart()
                        for i in range(self.dp_group_size):
                            for j, to_send in enumerate(server_data_compressed):
                                self.dp_comm.send(
                                    to_send, dst=i, stream=cupy_dp_stream)
                                # if j==0:
                                #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                                # if to_send.dtype == torch.uint8  and to_send.numel() > 1000:
                                #     _enc = _lossless_compress(to_send)
                                #     send_bits += len(_enc) * 8
                                # else:
                                #     send_bits += to_send.numel() * torch.finfo(to_send.dtype).bits if to_send.is_floating_point() else to_send.numel() * torch.iinfo(to_send.dtype).bits
                            for to_recv in comm_buffer_list[i]:
                                self.dp_comm.recv(
                                    to_recv, src=i, stream=cupy_dp_stream)
                                # recv_bits += to_recv.numel() * torch.finfo(to_recv.dtype).bits if to_recv.is_floating_point() else to_recv.numel() * torch.iinfo(to_recv.dtype).bits
                        cupy.cuda.nccl.groupEnd()
                        
                        # print('done')

                        for i in range(self.dp_group_size):
                            
                            _data = self._decompress(comm_buffer_list[i], comm_data_meta_list[i])
                            para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data * global_lr
                            global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] += _data * global_lr
                            
                            # para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]] = \
                            # global_para.data[i*chunk_size:(i+1)*chunk_size][comm_mask_list[i]].float()

                self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
                
                print('done partial sync')
                # print(f'param: {original_bits}, send: {send_bits}, recv: {recv_bits}')
                
#     def _copy_to_model(self):
        
#         if self.flatten:
#             name = 'model'
#             para = self.flatten_para
#             if name in self.comm_mask_dict:
#                 comm_mask = self.comm_mask_dict[name]
#                 buffer = self.buffer_dict[name]
#                 para.data[comm_mask] = buffer.to(para.dtype)
#         else:
#             for name, para in self.module.named_parameters():
#                 if name not in self.comm_mask_dict:
#                     continue
#                 comm_mask = self.comm_mask_dict[name]
#                 buffer = self.buffer_dict[name]
#                 para.data[comm_mask] = buffer.to(para.dtype)
            
    def optimizer_step(self):
        if flag.FLAG_DISABLE_COMPRESSION:
            self._allreduce_gradients()
        else:
            self._partial_sync()
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.sync_gradients_ready_event)
            self.torch_optim_comp_stream.wait_event(self.backward_ready_event)
            self.profile_mark_optimizer_step_start()
            # self._copy_to_model()
            self.optimizer.step()
            print('done optim')
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        # assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(self.sync_gradients_ready_event)*1e+3
        allreduce_log = {"name": "opt_shardedPS_sync", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.sync_gradients_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        
        
        allreduce_slot = self.gather_start_event.elapsed_time(self.gather_end_event)*1e+3
        allreduce_log = {"name": "gather grads", "ph": "X", "pid": self.global_rank, "tid": "9. optimizer-comm",
                         "ts": self.get_ts(self.gather_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.sync_start_event.elapsed_time(self.sync_end_event)*1e+3
        allreduce_log = {"name": "distribute grads", "ph": "X", "pid": self.global_rank, "tid": "10. optimizer-comm",
                         "ts": self.get_ts(self.sync_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.worker_compress_start_event.elapsed_time(self.worker_compress_end_event)*1e+3
        allreduce_log = {"name": "worker compress", "ph": "X", "pid": self.global_rank, "tid": "11. optimizer-comm",
                         "ts": self.get_ts(self.worker_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.server_compress_start_event.elapsed_time(self.server_compress_end_event)*1e+3
        allreduce_log = {"name": "server compress", "ph": "X", "pid": self.global_rank, "tid": "12. optimizer-comm",
                         "ts": self.get_ts(self.server_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        return profiling_log
