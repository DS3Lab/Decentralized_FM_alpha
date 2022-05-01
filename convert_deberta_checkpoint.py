import os
import argparse
import torch
from transformers import GPT2Model
# from modules.gpt_modules import GPTEmbeddings, GPTBlock



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='gpt2', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name)
    
    model = GPT2Model.from_pretrained(args.model_name)
    model.save_pretrained(save_path)

    torch.save({
        'wpe.weight': model.wpe.state_dict()['weight'],
        'wte.weight': model.wte.state_dict()['weight'],
    }, os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.h)):
        torch.save(model.h[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))