from .modules import *

def get_compressor(*args, compress_method='none', **kargs):
    if compress_method == 'none':
        return NoCompression(*args, **kargs)
    elif compress_method == 'fixpoint':
        return FixpointCompressor(*args, **kargs)
    else:
        raise Exception('unknown compression method')