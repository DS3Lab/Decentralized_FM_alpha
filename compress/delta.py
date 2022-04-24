import numpy as np
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

import concurrent.futures

def set_madvise(large_data, advise=1):
    '''
    0: MADV_NORMAL
    1: MADV_RANDOM
    2: MADV_SEQUENTIAL
    3: MADV_WILLNEED
    4: MADV_DONTNEED
    '''
    import ctypes
    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    assert madvise(large_data.ctypes.data, large_data.size * large_data.dtype.itemsize, advise) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM
    
    
class TensorCache(object):
    def __init__(self, shape, dtype=np.float16, tmp_path='/tmp/', device='cpu'):
        
        self.tmp_f = tempfile.NamedTemporaryFile(dir=tmp_path)
        
        self.cache_np = np.memmap(
            self.tmp_f, mode='w+', dtype=np.float16, shape=shape,
        )
        self.cache_torch = torch.from_numpy(self.cache_np)
        self.device = device
        
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.future_get = None
        self.future_set = None
        
        # 1: RANDOM ACCESS FLAG
        set_madvise(self.cache_np, 1)
        
    def get_tensor(self, k):
        return self.cache_torch[k].to(self.device)
    
    def set_tensor(self, k, v):
        v = v.detach().cpu().half()
        self.cache_torch[k] = v
        
    def launch_get_tensor(self, k):
        self.future_get = self.executor.submit(self.get_tensor, k)
    
    def wait_get_tensor(self):
        if self.future_get is not None:
            result = self.future_get.result()
            self.future_get = None
            return result
        else:
            raise Exception('no launched result')
    
    def launch_set_tensor(self, k, v):
        self.future = self.executor.submit(self.set_tensor, k, v)
        
    def wait_set_tensor(self):
        if self.future_set is not None:
            self.future_set.result()
            return
        else:
            raise Exception('no launched result')
        