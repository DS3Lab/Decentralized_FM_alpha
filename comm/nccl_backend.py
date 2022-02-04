import torch
import numpy as np
import cupy
import torch.distributed as dist
from typing import List


def _type_torch_to_cupy(torch_type: torch.dtype):
    # print(torch_type)
    mappings = {
        torch.uint8: cupy.cuda.nccl.NCCL_UINT8,
        torch.int32: cupy.cuda.nccl.NCCL_INT32,
        torch.int: cupy.cuda.nccl.NCCL_INT,
        torch.float16: cupy.cuda.nccl.NCCL_FLOAT16,
        torch.float32: cupy.cuda.nccl.NCCL_FLOAT32,
        torch.float64: cupy.cuda.nccl.NCCL_FLOAT64,
        torch.float: cupy.cuda.nccl.NCCL_FLOAT
    }
    return mappings[torch_type]


class NCCLCommunicator:
    def __init__(self,
                 comm_rank: int,
                 cuda_id: int,
                 comm_group_size: int,
                 comm_name: str):
        self.comm_rank = comm_rank
        cupy.cuda.Device(cuda_id).use()
        self.comm_group_size = comm_group_size
        print("Initialize NCCLCommunicator: <", comm_name, ">; rank:", comm_rank)
        dist_store = dist.distributed_c10d._get_default_store()

        if self.comm_rank == 0:
            cuda_id = cupy.cuda.nccl.get_unique_id()
            # print(cuda_id)
            cuda_id_str = np.array(cuda_id).tobytes()
            dist_store.set('group-'+comm_name+'-unique-id', cuda_id_str)
            # print("Master put <group-"+comm_name+"-unique-id: ", cuda_id_str, ">.")
        else:
            cuda_id_str = dist_store.get('group-'+comm_name+'-unique-id')

        comm_id = tuple(np.frombuffer(cuda_id_str, dtype=int))
        # comm_id = cupy.cuda.nccl.get_unique_id()
        # print(comm_id)
        self.comm = cupy.cuda.nccl.NcclCommunicator(comm_group_size, comm_id, comm_rank)

    @staticmethod
    def barrier():
        dist.barrier()

    def send(self,
             tensor: torch.Tensor,
             dst: int,
             stream=cupy.cuda.Stream.null):
        # print("Send tensor of size:", torch.numel(tensor))
        self.comm.send(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            dst,
            stream.ptr
        )

    def recv(self,
             tensor: torch.Tensor,
             src: int,
             stream=cupy.cuda.Stream.null):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        self.comm.recv(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            src,
            stream.ptr
        )

    def broadcast(self,
                  tensor: torch.Tensor,
                  src: int,
                  stream=cupy.cuda.Stream.null):
        self.comm.bcast(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            src,
            stream.ptr
        )

    def reduce(self,
               tensor: torch.Tensor,
               dst: int,
               stream=cupy.cuda.Stream.null,
               op=cupy.cuda.nccl.NCCL_SUM):
        self.comm.reduce(
            tensor.data_ptr(),  # force it to be in-place.
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            op,
            dst,
            stream.ptr
        )

    def all_reduce(self,
                  tensor: torch.Tensor,
                  stream=cupy.cuda.Stream.null,
                  op=cupy.cuda.nccl.NCCL_SUM):
        self.comm.allReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            op,
            stream.ptr
        )

    def scatter(self,
                tensor: torch.Tensor,
                scatter_list: List[torch.Tensor],
                src: int,
                stream=cupy.cuda.Stream.null):
        cupy.cuda.nccl.groupStart()
        if self.rank == src:
            for i in range(self.world_size):
                self.send(
                    scatter_list[i],
                    i,
                    stream
                )
        self.recv(
            tensor,
            src,
            stream
        )
        cupy.cuda.nccl.groupEnd()

    def gather(self,
               tensor: torch.Tensor,
               gather_list: List[torch.Tensor],
               dst: int,
               stream=cupy.cuda.Stream.null):
        cupy.cuda.nccl.groupStart()
        if self.rank == dst:
            for i in range(self.world_size):
                self.recv(
                    gather_list[i],
                    i,
                    stream
                )
        self.send(
            tensor,
            dst,
            stream
        )
        cupy.cuda.nccl.groupEnd()


def default_init(args):
    dist.init_process_group(backend='gloo', init_method=args.master_ip, world_size=args.pipeline_group_size, rank=args.rank)

"""
def init_comm(args):
    if args.dist_backend == 'cupy_nccl':
        comm = NCCLCommunicator(rank=args.rank, intra_gpu_rank=args.cuda_id,
                                world_size=args.world_size, master_ip=args.dist_url)
    else:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                rank=args.rank, world_size=args.world_size)
        comm = dist
    dist.barrier()
    return comm
"""