from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_1f1b_pipeline_async import Pipe1F1BAsync
from modules.dist_deberta_pp_module import *


def get_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    elif args.pp_mode == '1f1b':
        raise Exception('not implemented')
        return Pipe1F1BAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
def get_deberta_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    elif args.pp_mode == '1f1b':
        raise Exception('not implemented')
        return Pipe1F1BAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
