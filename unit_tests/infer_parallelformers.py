'''
Lauch inference with the following command:

$ python infer_parallelformers.py

'''

import torch
import time

from transformers import OPTForCausalLM, AutoTokenizer, OPTConfig

from parallelformers import parallelize

def main():

    batch_size = 1
    prompt_length = 512
    token_length = 50
    model_name_or_path = 'facebook/opt-125m'
    num_gpus = 1
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
            result = model.generate(input_ids, max_new_tokens=token_length, return_dict_in_generate=True, do_sample=True, output_scores=True)
            print(result)
            print('\n')
            print(tokenizer.decode(result['sequences'][0]))

        toc = time.time()

    print((toc - tic) / 5)


if __name__ == '__main__':
    main()
