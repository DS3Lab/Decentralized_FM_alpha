from .nccl_backend import *


_GPU_PIPELINE_COMM = None
_GPU_PIPELINE_RANK = None
_GPU_PIPELINE_WORLD_SIZE = None
_CPU_RANKS = None


def get_gpu_pipeline_comm() -> NCCLCommunicator:
    assert _GPU_PIPELINE_COMM is not None
    return _GPU_PIPELINE_COMM


def get_gpu_pipeline_rank() -> int:
    assert _GPU_PIPELINE_RANK is not None
    return _GPU_PIPELINE_RANK


def get_gpu_pipeline_world_size() -> int:
    assert _GPU_PIPELINE_WORLD_SIZE is not None
    return _GPU_PIPELINE_WORLD_SIZE


def get_cpu_ranks()-> List[int]:
    assert _CPU_RANKS is not None
    return _CPU_RANKS


def get_hybrid_dispatch_comm():
    return dist


def get_hybrid_dispatch_rank() -> int:
    return dist.get_rank()


def get_hybrid_dispatch_world_size() -> int:
    return dist.get_world_size()


def _init_hybrid_communicators(args):
    # world_size, pipeline_group_size, rank, cuda_id = 0
    assert args.world_size > args.pipeline_group_size
    dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    global _GPU_PIPELINE_COMM
    global _GPU_PIPELINE_RANK
    global _GPU_PIPELINE_WORLD_SIZE
    global _CPU_RANKS

    _GPU_PIPELINE_WORLD_SIZE = args.pipeline_group_size
    if args.rank < args.pipeline_group_size:
        _GPU_PIPELINE_RANK = args.rank
        _GPU_PIPELINE_COMM = NCCLCommunicator(_GPU_PIPELINE_RANK, args.cuda_id, args.pipeline_group_size,
                                              "pipeline_GPU_group")
    _CPU_RANKS = [i for i in range(args.pipeline_group_size, args.world_size)]

