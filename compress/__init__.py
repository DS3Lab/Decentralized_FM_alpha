from .dummy_modules import NoCompression
from .fixpoint_modules import FixpointCompressor, FixpointFlexibleCompressor
from .sparsification_modules import TopKCompressor
from .delta_modules import DeltaCompressor, DeltaLowBitsCompressor, DeltaTopKLowBitsCompressor, TopKDeltaCompressor
# from .simulate_modules import SimulateTestCompression, SimulateDeltaCompression

def get_compressor(*args, compress_method='none', **kargs):
    if compress_method == 'none':
        return NoCompression(*args, **kargs)
    elif compress_method == 'fixpoint':
        return FixpointFlexibleCompressor(*args, **kargs)
    elif compress_method == 'topk':
        return TopKCompressor(*args, **kargs)
    elif compress_method == 'delta':
        return DeltaCompressor(*args, **kargs)
    elif compress_method == 'delta-lowbits':
        return DeltaLowBitsCompressor(*args, **kargs)
    elif compress_method == 'delta-topk-lowbits':
        return DeltaTopKLowBitsCompressor(*args, **kargs)
    elif compress_method == 'topk-delta':
        return TopKDeltaCompressor(*args, **kargs)
    # elif compress_method == 'simulate':
    #     return SimulateTestCompression(*args, **kargs)
    # elif compress_method == 'simulate-delta':
    #     return SimulateDeltaCompression(*args, **kargs)
    else:
        raise Exception('unknown compression method')