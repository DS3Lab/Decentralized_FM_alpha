import torch
import cupy
import numpy as np
from torch.utils.dlpack import to_dlpack, from_dlpack

def _cupy_to_tensor(x):
    return from_dlpack(x.toDlpack())

def _tensor_to_cupy(x):
    return cupy.fromDlpack(to_dlpack(x))

def topr(x, ratio):
    x_flat = x.view(-1)
    numel = x.numel()
    k = max(int(numel * ratio), 1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.bool)
    masks[indexes] = 1
    masks = masks.view(x.shape)
    values = x.data[masks]
    return values, masks

def topk(x, k):
    x_flat = x.view(-1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.bool)
    masks[indexes] = 1
    masks = masks.view(x.shape)
    values = x.data[masks]
    return values, masks

def compress_topk(x, k):
    values, masks = topk(x, k)
    masks = _cupy_to_tensor(
        cupy.packbits(_tensor_to_cupy(masks))
    )
    return values, masks

def decompress_topk(values, masks, original_shape):
    masks = _cupy_to_tensor(
        cupy.unpackbits(_tensor_to_cupy(masks))
    )
    x = torch.zeros(masks.shape, dtype=values.dtype, device=values.device)
    x[masks] = values
    x = x.view(original_shape)
    return x