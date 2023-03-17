from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_gpipe_pipeline_sync import GpipeSync
from .dist_1f1b_pipeline_async import Pipe1F1BAsync
from .dist_gpipe_pipeline_async_act_comp import GpipeAsyncActivationCompression
from .dist_gpipe_pipeline_async_distill import GpipeAsyncDistill
from .dist_no_pipeline_async import NopipeAsync


def get_pp_module(args, config, device, use_dp):
    
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    elif args.pp_mode == 'nopipe':
        return NopipeAsync(args, config, device, use_dp)
    elif args.pp_mode == 'gpipe_sync':
        return GpipeSync(args, config, device, use_dp)
    elif args.pp_mode == '1f1b':
        raise Exception('not implemented')
        return Pipe1F1BAsync(args, config, device, use_dp)
    elif args.pp_mode == 'gpipe_act_comp':
        return GpipeAsyncActivationCompression(args, config, device, use_dp)
    elif args.pp_mode == 'gpipe_distill':
        return GpipeAsyncDistill(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
def get_deberta_pp_module(args, config, device, use_dp):
    
    from modules.dist_deberta_pp_module import DebertaStageFirst, DebertaStageLast, DebertaStageMiddle
    
    if args.pp_mode == 'gpipe':
        return GpipeAsync(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    elif args.pp_mode == 'gpipe_sync':
        return GpipeSync(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    elif args.pp_mode == '1f1b':
        raise Exception('not implemented')
        return Pipe1F1BAsync(args, config, device, use_dp)
    elif args.pp_mode == 'gpipe_act_comp':
        return GpipeAsyncActivationCompression(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
