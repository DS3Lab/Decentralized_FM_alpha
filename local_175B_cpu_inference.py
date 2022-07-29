#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
from time import time
import intel_extension_for_pytorch as ipex


def _create_layers(args, dtype=torch.float16):
    if args.model_type == 'gptj':
        from modules.hf_gptj_module import GPTBlock
    else:
        raise Exception(f'unknown model type {args.model_type}')
    cpu_layers = []
    for layer_index in range(args.num_layers):
        print(f'loading layer {layer_index}')
        current_layer = GPTBlock.from_pretrained(args.model_name, layer_index=layer_index, skip_ln=True).to(dtype).eval()
        current_layer = current_layer.to(memory_format=torch.channels_last)
        current_layer = ipex.optimize(current_layer)
        cpu_layers.append(current_layer)
    return cpu_layers


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--model-name', type=str, default='./pretrained_models/gpt-j-175B', metavar='S',
                        help='trained model path')
    parser.add_argument('--model-type', type=str, default='gptj', metavar='S',
                        help='trained model path')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--num-layers', type=int, default=96, metavar='N',
                        help='-')
    parser.add_argument('--prompt-seq-length', type=int, default=1024, metavar='N',
                        help='-')
    parser.add_argument('--gen-seq-length', type=int, default=100, metavar='N',
                        help='-')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    model = _create_layers(args, dtype=dtype)

    inputs = torch.empty((args.batch_size, args.prompt_seq_length, 12288),
                         requires_grad=False, dtype=dtype).normal_(mean=0.1, std=0.2)
    # inputs = inputs.to(memory_format=torch.channels_last)
    cached_tuples = [None for _ in range(args.num_layers)]

    with torch.no_grad():
        start_time = time()
        # prompt phase
        for layer_index in range(args.num_layers):
            if layer_index == 0:
                embeddings, cached_tuples[layer_index] = model[layer_index](inputs, skip_ln=True)
            else:
                embeddings, cached_tuples[layer_index] = model[layer_index](embeddings, skip_ln=True)

        prompt_end_time = time()
        print("Prompt phase takes {:3.2f}s".format(prompt_end_time-start_time))

        for i in range(args.gen_seq_length):
            inputs = torch.empty((args.batch_size, 1, 12288),
                                 requires_grad=False, dtype=dtype).normal_(mean=0.1, std=0.2)
            # inputs = inputs.to(memory_format=torch.channels_last)
            token_start_time = time()
            embeddings = None
            # print(inputs.shape)
            for layer_index in range(args.num_layers):
                if layer_index == 0:
                    embeddings, cached_tuples[layer_index] = model[layer_index](inputs, cached_tuples[layer_index],
                                                                                skip_ln=True)
                else:
                    embeddings, cached_tuples[layer_index] = model[layer_index](embeddings, cached_tuples[layer_index],
                                                                                skip_ln=True)
            token_end_time = time()
            print("Token <{}> takes {:3.2f}s".format(i, token_end_time - token_start_time))


if __name__ == '__main__':
    main()


