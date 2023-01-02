import torch
import torch.distributed as dist
from typing import List


class TorchCommunicator:
        
    def __init__(self,
                 process_group,
                 to_global_rank=lambda rank: rank):
        self.process_group = process_group
        self.to_global_rank = to_global_rank

    @staticmethod
    def barrier():
        dist.barrier()

    def send(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        dist.send(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)

    def recv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        buffer = tensor.cpu()
        dist.recv(buffer, self.to_global_rank(src), group=self.process_group)
        tensor[:] = buffer.to(tensor.device)
    
    def isend(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        handler = dist.isend(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)
        return handler

    def irecv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        # assert tensor is on CPU
        handler = dist.irecv(tensor, self.to_global_rank(src), group=self.process_group)
        return handler

    def broadcast(self,
                  tensor: torch.Tensor,
                  src: int,
                  stream=None):
        dist.broadcast(tensor, self.to_global_rank(src), group=self.process_group)

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
