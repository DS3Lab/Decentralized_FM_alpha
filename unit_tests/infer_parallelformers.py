'''
Lauch inference with the following command:

$ python infer_parallelformers.py

'''

import torch
import time

from transformers import OPTForCausalLM, AutoTokenizer, OPTConfig

from parallelformers import parallelize

def main():

    batch_size = 32
    prompt_length = 512
    token_length = 50
    model_name_or_path = 'facebook/opt-1.3b'
    num_gpus = 8
    fp16 = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = OPTConfig.from_pretrained(model_name_or_path)
    model = OPTForCausalLM(config)

    parallelize(model, num_gpus=num_gpus, fp16=fp16, verbose='detail')

    torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(5+1):
            if i == 1:
                # skip first
                tic = time.time()
            input_ids = tokenizer(['hello'] * batch_size, max_length=prompt_length, padding='max_length', return_tensors='pt')['input_ids'].cuda()
            model.generate(input_ids, max_new_tokens=token_length)

        toc = time.time()

    print((toc - tic) / 5)


if __name__ == '__main__':
    main()
