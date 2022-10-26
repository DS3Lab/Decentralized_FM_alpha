import os
import argparse
import torch
from transformers import GPT2Model, GPT2LMHeadModel, AutoConfig, AutoTokenizer
# from modules.gpt_modules import GPTEmbeddings, GPTBlock



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--input-path', type=str, default='model_checkpoints/gpt2-xl-prefix/checkpoint_1000', 
                        help='model-name')
    parser.add_argument('--output-path', type=str, default='model_checkpoints/gpt2-xl-prefix-result', 
                        help='model-name')
    parser.add_argument('--n-stages', type=int, default=2, 
                        help='')
    parser.add_argument('--n-layer-per-stage', type=int, default=14, 
                        help='')
    args = parser.parse_args()
    
    os.system(f'mkdir -p {args.output_path}')
    
    n_stages = args.n_stages
    n_layer_per_stage = args.n_layer_per_stage

    for i in range(n_stages):
        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))

            _tmp = {k[len(f"{n_layer_per_stage}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            if len(_tmp) == 0:
                break
            torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))