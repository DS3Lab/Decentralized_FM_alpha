
import torch
import cupy

def compress_nbit(x, bits, scale_method='max', scale_dims=(0,1)):
    
    fbits = bits - 1
    
    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax(scale_dims, keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean(scale_dims, keepdims=True).sqrt() * 2 
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    scale = scale.half()
    x = x / (scale + 1e-6)
    
    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = x.round()
    x = x.clip(clip_min, clip_max)
    
    x = x - clip_min
    x = x.type(torch.uint8)
    
    return x, scale


def decompress_nbits(x, scale, bits):
    
    fbits = bits - 1
    
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1
    
    x = x.float() + clip_min
    
    x = x / (clip_max+1) * scale
    
    return x


def compress_8bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = compress_nbit(x, bits=8, scale_method=scale_method, scale_dims=scale_dims)
    
    return x, scale


def decompress_8bit(x, scale):
    
    x = decompress_nbits(x, scale, bits=8)
    
    return x

def compress_4bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = compress_nbit(x, bits=4, scale_method=scale_method, scale_dims=scale_dims)
    
    x0, x1 = x.chunk(2, -1)
    x = (x0 << 4) + x1
    
    return x, scale


def decompress_4bit(x, scale):
    
    bitmask = 15
    
    x0 = (x >> 4)
    x1 = (x & bitmask)
    
    x = torch.cat([x0, x1], -1)
    
    x = decompress_nbits(x, scale, bits=4)
    
    return x


def compress_2bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = compress_nbit(x, bits=2, scale_method=scale_method, scale_dims=scale_dims)
    
    x0, x1, x2, x3 = x.chunk(4, -1)
    x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3
    
    return x, scale


def decompress_2bit(x, scale):
    
    bitmask = 3
    
    x0 = (x >> 6)
    x1 = (x >> 4) & bitmask
    x2 = (x >> 2) & bitmask
    x3 = x & bitmask
    x = torch.cat([x0, x1, x2, x3], -1)
    
    x = decompress_nbits(x, scale, bits=2)
    
    return x