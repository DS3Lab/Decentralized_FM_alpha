from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_gpipe_pipeline_async_offload import GpipeAsyncOffload
from .dist_1f1b_pipeline_async import Pipe1F1BAsync
from .dist_pipeline_inference_greedy import DistGreedyInferenceAsync
from .dist_pipeline_inference_greedy_sync import DistGreedyInferenceSync
from .dist_pipeline_inference_mask_greedy import DistGreedyInferenceMaskAsync
from .dist_pipeline_inference_mask_sample import DistSampleInferenceMaskAsync


def get_pp_module(args, vocab_size, num_classes, device, use_dp, rank=None):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp, rank=rank)
    # elif args.pp_mode =='gpipe_ao':
    #    return GpipeAsyncOffload(args, vocab_size, num_classes, device, use_dp, rank=rank)
    # elif args.pp_mode == '1f1b':
    #    return Pipe1F1BAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False


def get_pp_inference_module(args, device, rank=None):
    if args.pp_mode == 'pipe_async_greedy':
        return DistGreedyInferenceAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_greedy':
        return DistGreedyInferenceSync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_async_greedy_mask':
        return DistGreedyInferenceMaskAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_async_sample_mask':
        return DistSampleInferenceMaskAsync(args, device, rank=rank)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False