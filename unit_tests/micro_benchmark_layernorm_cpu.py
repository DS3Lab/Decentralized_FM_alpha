#!/usr/bin/env python
# coding: utf-8
import torch
import intel_extension_for_pytorch as ipex


def main():

    dtype = torch.bfloat16
    # dtype = torch.bfloat16 if args.fp16 else torch.float32
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim).to(dtype)

    layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=dtype).eval()
    layer_norm = layer_norm.to(memory_format=torch.channels_last)
    layer_norm = ipex.optimize(layer_norm)

    with torch.no_grad():
        with torch.autocast(device_type='cpu', dtype=dtype):
            output = layer_norm(embedding)
            print(output)



if __name__ == '__main__':
    main()


