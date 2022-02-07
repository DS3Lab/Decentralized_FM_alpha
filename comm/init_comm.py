from .nccl_backend import *

_DATA_PARALLEL_COMM = None
_PIPELINE_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = None
_PIPELINE_PARALLEL_RANK = None


def get_data_parallel_comm() -> NCCLCommunicator:
    assert _DATA_PARALLEL_COMM is not None
    return _DATA_PARALLEL_COMM


def get_pipeline_parallel_comm() -> NCCLCommunicator:
    assert _PIPELINE_PARALLEL_COMM is not None
    return _PIPELINE_PARALLEL_COMM


def get_data_parallel_rank() -> int:
    assert _DATA_PARALLEL_RANK is not None
    return _DATA_PARALLEL_RANK


def get_pipeline_parallel_rank() -> int:
    assert _PIPELINE_PARALLEL_RANK is not None
    return _PIPELINE_PARALLEL_RANK


def init_communicators(args):
    """
    We do the following hard code alignment of communication groups:
    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
    """
    assert args.pipeline_group_size == args.data_group_size * args.pipeline_group_size
    default_init(args)
    global _DATA_PARALLEL_COMM
    global _PIPELINE_PARALLEL_COMM
    global _DATA_PARALLEL_RANK
    global _PIPELINE_PARALLEL_RANK
    _PIPELINE_PARALLEL_RANK = args.rank % args.pipeline_group_size
    _DATA_PARALLEL_RANK = args.rank // args.pipeline_group_size
    # We use pipeline parallel by default.
    _PIPELINE_PARALLEL_COMM = NCCLCommunicator(_PIPELINE_PARALLEL_RANK, args.cuda_id, args.pipeline_group_size,
                                               "pipeline_group_"+str(args.rank // args.pipeline_group_size))
    if args.data_group_size != 1:
        _DATA_PARALLEL_RANK = NCCLCommunicator(_DATA_PARALLEL_RANK, args.cuda_id, args.data_group_size,
                                               "data_group_"+str(args.rank % args.pipeline_group_size))
