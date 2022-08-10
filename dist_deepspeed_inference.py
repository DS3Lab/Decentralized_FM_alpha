import os
import deepspeed
import argparse
import time
import torch
from utils.dist_args_utils import *
from transformers import OPTForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Deepspeed Inference-GPT3')

    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='rank of the node')
    parser.add_argument('--mp-size', type=int, default=8, help='size of tensor model parallelism')
    parser.add_argument('--dp-zero-stage', type=int, default=1, help='pipeline parallelism')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    device = torch.device('cuda', args.local_rank)

    checkpoint_json = {
    }
    # config = OPTConfig.from_pretrained('facebook/opt-125m')
    # model = OPTForCausalLM(config)
    model = OPTForCausalLM.from_pretrained('facebook/opt-125m')
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    #model = model_class.from_pretrained(args.model_name_or_path)
    # Initialize the DeepSpeed-Inference engine
    ds_engine = deepspeed.init_inference(model,
                                         mp_size=1,
                                         dtype=torch.half,
                                         checkpoint=None,
                                         replace_method='auto',
                                         replace_with_kernel_inject=True)
    text = "hello world!"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    model = ds_engine.module
    output = model(input_ids)
    print(output)

if __name__ == '__main__':
    main()