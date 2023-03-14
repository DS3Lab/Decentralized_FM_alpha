import os
import argparse
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modules.llama_modules import LLaMATokenizer, LLaMAForCausalLM, LLaMAConfig

DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='EleutherAI/gpt-neox-20b', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default=DIR, 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config = LLaMAConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_path)
    tokenizer = LLaMATokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    model = LLaMAForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    
    item = {}
    item['embed_tokens.weight'] = model.model.embed_tokens.weight
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.model.layers)):
        torch.save(model.model.layers[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
    
    item = {}
    item['lm_head.weight'] = model.lm_head.weight
    item['norm.weight'] = model.model.norm.weight
    torch.save(item, os.path.join(save_path, 'pytorch_lm_head.pt'))