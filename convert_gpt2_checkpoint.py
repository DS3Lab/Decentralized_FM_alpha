import os
import argparse
import torch
from transformers import GPT2Model, GPT2LMHeadModel, AutoConfig, AutoTokenizer
# from modules.gpt_modules import GPTEmbeddings, GPTBlock



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='gpt2', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default='./', 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name)
    os.system(f"mkdir -p {save_path}")
    
    AutoTokenizer.from_pretrained(args.model_name).save_pretrained(save_path)
    AutoConfig.from_pretrained(args.model_name).save_pretrained(save_path)

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
#     model.save_pretrained(save_path)

    torch.save({
        'wpe.weight': model.transformer.wpe.state_dict()['weight'],
        'wte.weight': model.transformer.wte.state_dict()['weight'],
    }, os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.transformer.h)):
        torch.save(model.transformer.h[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
        
    torch.save({
        'ln_f.weight': model.transformer.ln_f.state_dict()['weight'],
        'ln_f.bias': model.transformer.ln_f.state_dict()['bias'],
        'lm_head.weight': model.lm_head.state_dict()['weight'],
    }, os.path.join(save_path, 'pytorch_lm_head.pt'))
