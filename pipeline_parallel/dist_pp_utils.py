from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_1f1b_pipeline_async import Pipe1F1BAsync


def get_pp_module(args, vocab_size, num_classes, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp)
    elif args.pp_mode == '1f1b':
        return Pipe1F1BAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
