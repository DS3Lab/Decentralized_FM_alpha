import torch
import torch.distributed as dist
from typing import List
from compress.fixpoint import *
from compress.sparsification import *

class TorchCommunicator:
        
    def __init__(self,
                 process_group,
                 to_global_rank=lambda rank: rank,
                 dp_rank=None,
                 comm_group_size=None,):
        self.process_group = process_group
        self.to_global_rank = to_global_rank
        self.dp_rank = dp_rank
        self.comm_group_size = comm_group_size

    # @staticmethod
    def barrier(self):
        dist.barrier(group=self.process_group)

    def send(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        if tensor.device == torch.device('cpu'):
            dist.send(tensor, self.to_global_rank(dst), group=self.process_group)
        else:
            dist.send(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)
            
    def recv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        # buffer = tensor.cpu()
        # dist.recv(buffer, self.to_global_rank(src), group=self.process_group)
        # tensor[:] = buffer.to(tensor.device)
        
        if tensor.device == torch.device('cpu'):
            dist.recv(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            buffer = tensor.cpu()
            dist.recv(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)
    
    def isend(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        if tensor.device == torch.device('cpu'):
            handler = dist.isend(tensor, self.to_global_rank(dst), group=self.process_group)
        else:
            handler = dist.isend(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)
        return handler

    def irecv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        # assert tensor is on CPU
        if tensor.device == torch.device('cpu'):
            handler = dist.irecv(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            assert False
            buffer = tensor.cpu()
            handler = dist.irecv(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)
        return handler

    def broadcast(self,
                  tensor: torch.Tensor,
                  src: int,
                  stream=None):
        if tensor.device == torch.device('cpu'):
            dist.broadcast(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            buffer = tensor.cpu()
            dist.broadcast(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)

    def reduce(self,
               tensor: torch.Tensor,
               dst: int,
               stream=None,
               op=dist.ReduceOp.SUM):
        dist.reduce(tensor, self.to_global_rank(dst), group=self.process_group, op=op)

    def all_reduce(self,
                   tensor: torch.Tensor,
                   stream = None,
                   op=dist.ReduceOp.SUM):
        buffer = tensor.cpu()
        dist.all_reduce(buffer, group=self.process_group, op=op)
        tensor[:] = buffer.to(tensor.device)

    def gather(self,
               tensor: torch.Tensor,
               gather_list: List[torch.Tensor],
               dst: int,
               stream=None):
        dist.gather(tensor, gather_list, self.to_global_rank(dst), group=self.process_group)

    def all_to_all(self,
                   output_tensor_list: List[torch.Tensor],
                   input_tensor_list: List[torch.Tensor],
                   stream=None):
        dist.all_to_all(output_tensor_list, input_tensor_list, group=self.process_group)

    def all_gather(self,
                   tensor: torch.Tensor,
                   output_tensor_list: List[torch.Tensor],
                   stream=None):
        dist.all_gather(output_tensor_list, tensor, group=self.process_group)
        
    
    def all_reduce_opt_compressed(self,
                       tensor: torch.Tensor,
                       buffer: List[torch.Tensor],
                       worker_errors: List[torch.Tensor],
                       server_error: torch.Tensor,
                       stream=cupy.cuda.Stream.null,
                       bits=8, caller=None):
        
        if buffer[0][0].device != torch.device('cpu'):
            for i in range(len(buffer)):
                buffer[i] = [z.cpu() for z in buffer[i]]
                
        with stream:
            # First do all-to-all
            assert torch.numel(tensor.data) % self.comm_group_size == 0

            tensor_chunks = tensor.data.chunk(self.comm_group_size, 0)
            # all chunks have the same shape
            original_shape = tensor_chunks[0].shape
            
            for i in range(self.comm_group_size):
                worker_errors[i] = worker_errors[i].nan_to_num()
            server_error = server_error.nan_to_num()
            
#             caller.dp_comm_stream.record_event(caller.worker_compress_start_event)
            # worker error compensation
            for i in range(self.comm_group_size):
                tensor_chunks[i].add_(worker_errors[i])
            
            # decompress
#             tensor_chunks_compressed = [compress_flexible_nbits(
#                 _data, bits=bits, scale_dims=tuple()) for _data in tensor_chunks]
            tensor_chunks_compressed = [compress_flexible_nbits_by_bucket(
                _data, bits=bits, bucket_size=128) for _data in tensor_chunks]
            
            # update worker errors
            for i in range(self.comm_group_size):
                worker_errors[i].set_((tensor_chunks[i] - decompress_flexible_nbits_by_bucket(
                    *tensor_chunks_compressed[i], bits=bits, 
                    original_shape=original_shape, bucket_size=128)).type(worker_errors[i].dtype))
            del tensor_chunks
#             caller.dp_comm_stream.record_event(caller.worker_compress_end_event)
            
#             caller.dp_comm_stream.record_event(caller.gather_start_event)
            _group_calls = []
            for i in range(self.comm_group_size):
                for j, to_send in enumerate(tensor_chunks_compressed[i]):
                    if i != self.dp_rank:
                        # print('s', i, to_send.numel())
                        call = self.isend(
                            to_send, dst=i, stream=stream)
                        _group_calls.append(call)
                    else:
                        buffer[i][j][:] = to_send
                for to_recv in buffer[i]:
                    if i != self.dp_rank:
                        # print('r', i, to_recv.numel())
                        call = self.irecv(
                            to_recv, src=i, stream=stream)
                        _group_calls.append(call)
            for call in _group_calls:
                call.wait()
                
            self.barrier()
#             caller.dp_comm_stream.record_event(caller.gather_end_event)

#             caller.dp_comm_stream.record_event(caller.server_compress_start_event)
            tensor_server = decompress_flexible_nbits_by_bucket(
                *buffer[0], bits=bits, original_shape=original_shape, bucket_size=128)
            for i in range(1, self.comm_group_size):
                tensor_server += decompress_flexible_nbits_by_bucket(
                    *buffer[i], bits=bits, original_shape=original_shape, bucket_size=128)
            # tensor_server.mul_(1 / self.comm_group_size)
                
            # server error compensation
            tensor_server.add_(server_error)
            
#             tensor_server_compressed = compress_flexible_nbits(tensor_server, bits=bits, scale_dims=tuple())
            tensor_server_compressed = compress_flexible_nbits_by_bucket(tensor_server, bits=bits, bucket_size=128)
            
            # update server error
            server_error.set_((tensor_server - decompress_flexible_nbits_by_bucket(
                    *tensor_server_compressed, bits=bits, 
                    original_shape=original_shape, bucket_size=128)).type(server_error.dtype))
#             caller.dp_comm_stream.record_event(caller.server_compress_end_event)
            
#             caller.dp_comm_stream.record_event(caller.sync_start_event)
            _group_calls = []
            for i in range(self.comm_group_size):
                for j, to_send in enumerate(tensor_server_compressed):
                    if i != self.dp_rank:
                        call = self.isend(
                            to_send, dst=i, stream=stream)
                        _group_calls.append(call)
                    else:
                        buffer[i][j][:] = to_send
                for to_recv in buffer[i]:
                    if i != self.dp_rank:
                        call = self.irecv(
                            to_recv, src=i, stream=stream)
                        _group_calls.append(call)
            for call in _group_calls:
                call.wait()
                
            self.barrier()
#             caller.dp_comm_stream.record_event(caller.sync_end_event)

#             recv_tensors = [decompress_flexible_nbits_by_bucket(*_data, bits=bits, original_shape=original_shape, bucket_size=128) for _data in buffer]
#             tensor.data.copy_(torch.cat(recv_tensors, 0))
            for i, _data in enumerate(buffer):
                tensor.data[i*original_shape[0]:(i+1)*original_shape[0]] = \
                    decompress_flexible_nbits_by_bucket(*_data, bits=bits, original_shape=original_shape, bucket_size=128)

        
    def all_reduce_opt_topk(self,
                       tensor: torch.Tensor,
                       buffer: List[torch.Tensor],
                       worker_errors: List[torch.Tensor],
                       server_error: torch.Tensor,
                       stream=cupy.cuda.Stream.null,
                       topk_ratio=0.1, caller=None):
        
        if buffer[0][0].device != torch.device('cpu'):
            for i in range(len(buffer)):
                buffer[i] = [z.cpu() for z in buffer[i]]
        
        with stream:
            
            self.barrier()
            
            # First do all-to-all
            assert torch.numel(tensor.data) % self.comm_group_size == 0
            
            for i in range(self.comm_group_size):
                worker_errors[i] = worker_errors[i].nan_to_num()
            server_error = server_error.nan_to_num()

            tensor_chunks = tensor.data.chunk(self.comm_group_size, 0)
            # all chunks have the same shape
            original_shape = tensor_chunks[0].shape
            
#             caller.dp_comm_stream.record_event(caller.worker_compress_start_event)
            # worker error compensation
            for i in range(self.comm_group_size):
                tensor_chunks[i].add_(worker_errors[i])
            
            # decompress
            tensor_chunks_compressed = [compress_topk(
                _data, int(topk_ratio * _data.numel())) for _data in tensor_chunks]
            
            # update worker errors
            for i in range(self.comm_group_size):
                worker_errors[i].set_((tensor_chunks[i] - decompress_topk(
                    *tensor_chunks_compressed[i],
                    original_shape=original_shape)).type(worker_errors[i].dtype))
            del tensor_chunks
            # caller.dp_comm_stream.record_event(caller.worker_compress_end_event)

            # caller.dp_comm_stream.record_event(caller.gather_start_event)

            _group_calls = []
            for i in range(self.comm_group_size):
                for j, to_send in enumerate(tensor_chunks_compressed[i]):
                    if i != self.dp_rank:
                        # print('s', i, to_send.numel())
                        call = self.isend(
                            to_send, dst=i, stream=stream)
                        _group_calls.append(call)
                    else:
                        buffer[i][j][:] = to_send
                for to_recv in buffer[i]:
                    if i != self.dp_rank:
                        # print('r', i, to_recv.numel())
                        call = self.irecv(
                            to_recv, src=i, stream=stream)
                        _group_calls.append(call)
            for call in _group_calls:
                call.wait()
                
            self.barrier()

            # cupy.cuda.nccl.groupStart()
            # for i in range(self.comm_group_size):
            #     to_send = tensor_chunks_compressed[i][0]
            #     self.comm.send(
            #         to_send.data_ptr(), to_send.numel(), 
            #         _type_torch_to_cupy(to_send.dtype), i, stream.ptr)
            #     to_send = tensor_chunks_compressed[i][1]
            #     self.comm.send(
            #         to_send.data_ptr(), to_send.numel(), 
            #         _type_torch_to_cupy(to_send.dtype), i, stream.ptr)
            #     to_recv = buffer[i][0]
            #     self.comm.recv(
            #         to_recv.data_ptr(), to_recv.numel(),
            #         _type_torch_to_cupy(to_recv.dtype), i, stream.ptr)
            #     to_recv = buffer[i][1]
            #     self.comm.recv(
            #         to_recv.data_ptr(), to_recv.numel(),
            #         _type_torch_to_cupy(to_recv.dtype), i, stream.ptr)
            # cupy.cuda.nccl.groupEnd()

            # caller.dp_comm_stream.record_event(caller.gather_end_event)

            # caller.dp_comm_stream.record_event(caller.server_compress_start_event)
            tensor_server = decompress_topk(
                buffer[0][0].to(tensor.device), buffer[0][1].to(tensor.device), original_shape=original_shape,)
            for i in range(1, self.comm_group_size):
                tensor_server += decompress_topk(
                    buffer[i][0].to(tensor.device), buffer[i][1].to(tensor.device), original_shape=original_shape,)
            # tensor_server.mul_(1 / self.comm_group_size)

            # server error compensation
            tensor_server.add_(server_error)

            tensor_server_compressed = compress_topk(tensor_server, int(topk_ratio * tensor_server.numel()))

            # update server error
            server_error.set_((tensor_server - decompress_topk(
                    *tensor_server_compressed, 
                    original_shape=original_shape)).type(server_error.dtype))
            # caller.dp_comm_stream.record_event(caller.server_compress_end_event)

            # caller.dp_comm_stream.record_event(caller.sync_start_event)
            _group_calls = []
            for i in range(self.comm_group_size):
                for j, to_send in enumerate(tensor_server_compressed):
                    if i != self.dp_rank:
                        call = self.isend(
                            to_send, dst=i, stream=stream)
                        _group_calls.append(call)
                    else:
                        buffer[i][j][:] = to_send
                for to_recv in buffer[i]:
                    if i != self.dp_rank:
                        call = self.irecv(
                            to_recv, src=i, stream=stream)
                        _group_calls.append(call)
            for call in _group_calls:
                call.wait()
                
            self.barrier()

            # cupy.cuda.nccl.groupStart()
            # for i in range(self.comm_group_size):
            #     self.comm.send(
            #         tensor_server_compressed[0].data_ptr(), tensor_server_compressed[0].numel(), 
            #         _type_torch_to_cupy(tensor_server_compressed[0].dtype), i, stream.ptr)
            #     self.comm.send(
            #         tensor_server_compressed[1].data_ptr(), tensor_server_compressed[1].numel(), 
            #         _type_torch_to_cupy(tensor_server_compressed[1].dtype), i, stream.ptr)
            #     to_recv = buffer[i][0]
            #     self.comm.recv(
            #         to_recv.data_ptr(), to_recv.numel(),
            #         _type_torch_to_cupy(to_recv.dtype), i, stream.ptr)
            #     to_recv = buffer[i][1]
            #     self.comm.recv(
            #         to_recv.data_ptr(), to_recv.numel(),
            #         _type_torch_to_cupy(to_recv.dtype), i, stream.ptr)
            # cupy.cuda.nccl.groupEnd()
            # caller.dp_comm_stream.record_event(caller.sync_end_event)

            for i, _data in enumerate(buffer):
                tensor.data[i*original_shape[0]:(i+1)*original_shape[0]] = \
                    decompress_topk(_data[0].to(tensor.device), _data[1].to(tensor.device), original_shape=original_shape)


def default_init(args):
    if hasattr(args, 'world_size'):
        dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    else:
        assert hasattr(args, 'pipeline_group_size')
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.pipeline_group_size, rank=args.rank)


def init_with_coordinator(args, prime_ip, rank, port=9999):
    if hasattr(args, 'world_size'):
        dist.init_process_group(backend='gloo', init_method='tcp://'+prime_ip+f':{port}',
                                world_size=args.world_size, rank=rank)
    else:
        assert hasattr(args, 'pipeline_group_size')
        dist.init_process_group(backend='gloo', init_method='tcp://' + prime_ip + f':{port}',
                                world_size=args.pipeline_group_size, rank=rank)
