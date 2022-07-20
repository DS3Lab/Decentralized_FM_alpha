import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, batch_size=None):
        
        self.tokenizer = tokenizer
        self.data = data
        self.idx = list(range(len(data)))
        
        if batch_size is not None:
            n_dummy = batch_size - len(data) % batch_size
            if n_dummy < batch_size:
                self.idx += [-1]*n_dummy
                self.data = self.data + [tokenizer.bos_token]*n_dummy
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.tokenizer(
            self.data[idx], return_tensors='pt', 
            padding='max_length', truncation=True,
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item['text'] = item['input_ids']
        item['idx'] = self.idx[idx]

        return item
    
class DummyRequestProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def set_arguments(self, args):
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.temperature = args.temperature
        self.echo_prompt = args.echo_prompt
        self.top_k_per_token = args.top_k_per_token
        self.num_completions = args.num_completions
        self.max_tokens = args.generate_seq_length
        self.tokenizer.model_max_length = min(args.input_seq_length, self.tokenizer.model_max_length - args.generate_seq_length)
    
    def get_dataloader(self, batch_size, num_workers=0):
        if not self.echo_prompt:
            dataset = JsonDataset(
                ['you are not a']*2000, 
                self.tokenizer, batch_size=batch_size,
            )
        else:
            dataset = JsonDataset(
                [self.tokenizer.bos_token+'you are not a']*2000, 
                self.tokenizer, batch_size=batch_size,
            )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=None
        )
        
        return data_loader
        
    def add_result(self, inputs, outputs, batch_time=None):
        batch_size = len(inputs['idx'])
        tokenizer = self.tokenizer
        for i in range(batch_size):
            idx = inputs['idx'][i]
            if idx < 0:
                continue
                
            if self.echo_prompt:
                n_pads = (1-inputs['attention_mask'][i]).sum()
            else:
                n_pads = 0
                
            item = {
                'choices': [], 
                'request_time': {
                'batch_time': batch_time,
                'batch_size': batch_size,
                }
            }
            
            for i_ret, output_dict in enumerate(outputs):
                choice = {
                    "text": (tokenizer.decode(output_dict['token_ids'][i][n_pads:]).replace(tokenizer.pad_token, '') if 'token_ids' in output_dict else ''),
                    "index": i_ret,
                    "logprobs": {
                        "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i][n_pads:] if 'token_ids' in output_dict else [])),
                        "token_logprobs": (output_dict['token_logprobs'][i][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
                        "top_logprobs": ([
                            {
                                tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                            } \
                            for topk_ids, top_logprobs in zip(
                                output_dict['topk_ids'][i][n_pads:],
                                output_dict['topk_logprobs'][i][n_pads:]
                            )
                        ] if self.top_k_per_token > 0 and self.max_tokens > 0 else None),
                        "text_offset": [],
                    },
                    "finish_reason": "length",
                }
                item['choices'].append(choice)
            print(json.dumps(item, indent=4))
        
    def write_scenario_state(self):
        pass
    

class RequestProcessor:
    def __init__(self, request_path, tokenizer):
        
        self.tokenizer = tokenizer
        self.request_path = request_path
        dirname = os.path.dirname(request_path)
        basename = os.path.basename(request_path)
        self.output_path = os.path.join(dirname, 'output_'+basename)
        with open(self.request_path) as f:
            self.data = []
            for line in f:
                if line.strip() != '':
                    self.data.append({'request': json.loads(line)})
        first_request = self.data[0]['request']
        self.top_k = first_request.get('top_k', None)
        self.top_p = first_request.get('top_p', None)
        self.temperature = first_request.get('temperature', 1)
        self.echo_prompt = first_request.get('echo', 1)
        self.top_k_per_token = first_request.get('logprobs', 1)
        self.num_completions = first_request.get('n', 1)
        self.max_tokens = first_request.get('max_tokens', 1)
        self.best_of = first_request.get('best_of', 1)
        
    def set_arguments(self, args):
        if hasattr(args, 'overwrite_request_args') and args.overwrite_request_args:
            self.top_k = args.top_k
            self.top_p = args.top_p
            self.temperature = args.temperature
            self.echo_prompt = args.echo_prompt
            self.top_k_per_token = args.top_k_per_token
            self.num_completions = args.num_completions
            self.max_tokens = args.generate_seq_length
            self.best_of = args.best_of
        else:
            args.top_k = self.top_k
            args.top_p = self.top_p
            args.temperature = self.temperature
            args.echo_prompt = self.echo_prompt
            args.top_k_per_token = self.top_k_per_token
            args.num_completions = self.num_completions
            args.generate_seq_length = self.max_tokens
            args.best_of = self.best_of
        
            max_input_seq_length = 1
            for i, x in enumerate(self.data):
                seq_length = len(self.tokenizer(x['request']['prompt'])['input_ids'])
                if seq_length > max_input_seq_length:
                    max_input_seq_length = seq_length
                if i > 0:
                    break
            args.input_seq_length = min(
                max_input_seq_length + 30, 
                self.tokenizer.model_max_length - args.generate_seq_length,
            )
            
        self.tokenizer.model_max_length = args.input_seq_length

        print('input seq length:', args.input_seq_length)
        
    def get_dataloader(self, batch_size, num_workers=0):
        
        if not self.echo_prompt:
            dataset = JsonDataset(
                [x['request']['prompt'] for x in self.data], 
                self.tokenizer, batch_size=batch_size,
            )
        else:
            dataset = JsonDataset(
                [self.tokenizer.bos_token+x['request']['prompt'] for x in self.data], 
                self.tokenizer, batch_size=batch_size,
            )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False, # make it explicit
            pin_memory=True,
            collate_fn=None
        )
        
        return data_loader
    
    def add_result(self, inputs, outputs, batch_time=None):
        batch_size = len(inputs['idx'])
        tokenizer = self.tokenizer
        for i in range(batch_size):
            idx = inputs['idx'][i]
            if idx < 0:
                continue
                
            if self.echo_prompt:
                n_pads = (1-inputs['attention_mask'][i]).sum()
            else:
                n_pads = 0
                
            item = {
                'choices': [], 
                'request_time': {
                'batch_time': batch_time,
                'batch_size': batch_size,
                }
            }
            
            for i_ret, output_dict in enumerate(outputs):
                choice = {
                    "text": (tokenizer.decode(output_dict['token_ids'][i][n_pads:]).replace(tokenizer.pad_token, '') if 'token_ids' in output_dict else ''),
                    "index": i_ret,
                    "logprobs": {
                        "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i][n_pads:] if 'token_ids' in output_dict else [])),
                        "token_logprobs": (output_dict['token_logprobs'][i][n_pads:].tolist() if 'token_logprobs' in output_dict else []),
                        "top_logprobs": ([
                            {
                                tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                            } \
                            for topk_ids, top_logprobs in zip(
                                output_dict['topk_ids'][i][n_pads:],
                                output_dict['topk_logprobs'][i][n_pads:]
                            )
                        ] if self.top_k_per_token > 0 and self.max_tokens > 0 else None),
                        "text_offset": [],
                    },
                    "finish_reason": "length",
                }
                item['choices'].append(choice)
            self.data[idx]['result'] = item
            
            try:
                if self.num_completions > 1:
                    self.data[idx]['result']['choices'] = sorted(
                        self.data[idx]['result']['choices'],
                        key=lambda c: -np.mean(c['logprobs']['token_logprobs']),
                    )
                    self.data[idx]['result']['choices'][:self.best_of]
                    for _i, c in enumerate(self.data[idx]['result']['choices']):
                        c['index'] = _i
            except:
                print('fail to sort choices')
        
    def write_scenario_state(self):
        with open(self.output_path, 'w') as f:
            for line in self.data:
                f.write(json.dumps(line) + '\n')
            
            
def get_request_processor(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.model_type in ['t5']:
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
    else:
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
    # tokenizer.model_max_length = args.input_seq_length
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.infer_data.strip() == '':
        return DummyRequestProcessor(tokenizer)
    else:
        return RequestProcessor(args.infer_data, tokenizer)
