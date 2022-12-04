
import torch
import torch.nn as nn

from transformers import OPTForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights

import json

import os


def create_emtpy_opt(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = OPTForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=3, n_layer_per_stage=12, max_num_layers=24):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.model.decoder.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.model.decoder.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']
            model.model.decoder.embed_positions.weight.data[:] = _tmp['embed_positions.weight']

            for j in range(n_layer_per_stage):
                
                print('loading', i*n_layer_per_stage + j, '...')
                
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.model.decoder.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                
                if i*n_layer_per_stage + j >= max_num_layers:
                    break
                    
                print('loading', i*n_layer_per_stage + j, '...')
                
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.decoder.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

            print('loading final layer...')
            
            _tmp = {k[len(f"{n_layer_per_stage}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            assert len(_tmp) > 0
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.model.decoder.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.model.decoder.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                
                print('loading', i*n_layer_per_stage + j, '...')
                
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.decoder.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


def infer(prompt, max_new_tokens=1):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {
            k: v.to(model.device) for k,v in inputs.items()
        }
        ret = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

        return tokenizer.decode(ret[0, inputs['input_ids'].size(1):])
    
    
from sklearn.metrics import classification_report

def evaluate_acc(task_path, max_new_tokens=20, k_shot=-1, first_100=True):
    
    with open(task_path) as f:
        state_dict = json.load(f)
    
    n_c = 0
    n_t = 0
    for request in state_dict['request_states']:
        
        if 'perturbation' in request['instance']:
            continue
        
        prompt = request['request']['prompt']
        label = request['instance']['references'][0]['output']
        pred = infer(prompt, max_new_tokens=max_new_tokens).strip().split('\n')[0]
        
        if label == pred:
            n_c += 1
        n_t += 1
        
    return n_c / n_t


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", help="ckpts")
    args = parser.parse_args()
    
    ckpts = os.listdir(args.ckpt_root)
    ckpt_paths = [os.path.join(args.ckpt_root, ckpt) for ckpt in sorted(ckpts)]
    
    config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    
    model = create_emtpy_opt(config).half().eval()
    model = model.to('cuda:0')
    
    for ckpt_path in ckpt_paths:
        load_decentralized_checkpoint(model, ckpt_path, n_stages=3, n_layer_per_stage=12, max_num_layers=24)
        
        # for i in range(24):
        #     model.transformer.h[i].attn.bias.data[:] = 1
        
        item = {
            'ckpt_path': ckpt_path,
        }
        acc_list = []
        
        tasks_list = [
            'ade_corpus_v2',
            'banking_77',
            'neurips_impact_statement_risks',
            'one_stop_english',
            'overruling',
            'semiconductor_org_types',
            'systematic_review_inclusion',
            'tai_safety_research',
            'terms_of_service',
            'tweet_eval_hate',
            'twitter_complaints',
        ]
        
        for task_name in tasks_list:
            acc = evaluate_acc(f'data/raft/raft:subset={task_name},model=together_gpt-j-6b,data_augmentation=canonical/scenario_state.json', max_new_tokens=20)
            acc_list.append(acc)
            item[f"raft:{task_name}"] = acc
            print(task_name, acc)
            
        acc = sum(acc_list) / len(acc_list)
        item['raft'] = acc
        
        with open('result.jsonl', 'a') as fout:
            fout.write(json.dumps(item) + '\n')
        