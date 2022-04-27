from .dummy_modules import NoCompression
from .fixpoint_modules import FixpointCompressor, FixpointFlexibleCompressor
from .sparsification_modules import TopKCompressor
from .delta_modules import DeltaCompressor, DeltaLowBitsCompressor

def get_compressor(*args, compress_method='none', bits=8, **kargs):
    if compress_method == 'none':
        return NoCompression(*args, bits=bits, **kargs)
    elif compress_method == 'fixpoint':
        if bits in {2, 4, 8}:
            return FixpointCompressor(*args, bits=bits, **kargs)
        else:
            return FixpointFlexibleCompressor(*args, bits=bits, **kargs)
    elif compress_method == 'topk':
        return TopKCompressor(*args, **kargs)
    elif compress_method == 'delta':
        return DeltaCompressor(*args, bits=bits, **kargs)
    elif compress_method == 'delta-lowbits':
        return DeltaLowBitsCompressor(*args, bits=bits, **kargs)
    else:
        raise Exception('unknown compression method')