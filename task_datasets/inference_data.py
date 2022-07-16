import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, batch_size=None):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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
    
    def get_dataloader(self, batch_size, num_workers=0):
        dataset = JsonDataset(
            ['you are']*2000, 
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
                
            echo_text = tokenizer.decode(
                inputs['text'][i]
            ).replace('<|endoftext|>', '')
            item = {
                'choices': [
                    {
                        "text": (echo_text + (tokenizer.decode(output_dict['token_ids'][i]).replace('<|endoftext|>', '') if 'token_ids' in output_dict else '')),
                        "index": i_ret,
                        "logprobs": {
                            "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i] if 'token_ids' in output_dict else [])),
                            "token_logprobs": (output_dict['token_logprobs'][i].tolist() if 'token_logprobs' in output_dict else []),
                            "top_logprobs": ([
                                {
                                    tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                                } \
                                for topk_ids, top_logprobs in zip(
                                    output_dict['topk_ids'][i], 
                                    output_dict['topk_logprobs'][i]
                                )
                            ] if self.top_k_per_token > 0 and self.max_tokens > 0 else None),
                            "text_offset": [],
                        },
                        "finish_reason": "stop",
                    } for i_ret, output_dict in enumerate(outputs)
                ],
                'request_time': {
                    'batch_time': batch_time,
                    'batch_size': batch_size,
                }
            }
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
        
    def set_arguments(self, args):
        if hasattr(args, 'overwrite_request_args') and args.overwrite_request_args:
            return
        args.top_k = self.top_k
        args.top_p = self.top_p
        args.temperature = self.temperature
        args.echo_prompt = self.echo_prompt
        args.top_k_per_token = self.top_k_per_token
        args.num_completions = self.num_completions
        args.generate_seq_length = self.max_tokens
        
    def get_dataloader(self, batch_size, num_workers=0):
        
        dataset = JsonDataset(
            [x['request']['prompt'] for x in self.data], 
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
                echo_text = tokenizer.decode(
                    inputs['text'][i]
                ).replace('<|endoftext|>', '')
            else:
                echo_text = ''
            item = {
                'choices': [
                    {
                        "text": (echo_text + (tokenizer.decode(output_dict['token_ids'][i]).replace('<|endoftext|>', '') if 'token_ids' in output_dict else '')),
                        "index": i_ret,
                        "logprobs": {
                            "tokens": (tokenizer.convert_ids_to_tokens(output_dict['token_ids'][i] if 'token_ids' in output_dict else [])),
                            "token_logprobs": (output_dict['token_logprobs'][i].tolist() if 'token_logprobs' in output_dict else []),
                            "top_logprobs": ([
                                {
                                    tokenizer.convert_ids_to_tokens(topk_id.item()): top_logprob.item()  for topk_id, top_logprob in zip(topk_ids, top_logprobs)
                                } \
                                for topk_ids, top_logprobs in zip(
                                    output_dict['topk_ids'][i], 
                                    output_dict['topk_logprobs'][i]
                                )
                            ] if self.top_k_per_token > 0 and self.max_tokens > 0 else None),
                            "text_offset": [],
                        },
                        "finish_reason": "length",
                    } for i_ret, output_dict in enumerate(outputs)
                ],
                'request_time': {
                    'batch_time': batch_time,
                    'batch_size': batch_size,
                }
            }
            self.data[idx]['result'] = item
            # print(json.dumps(item, indent=4))
        
    def write_scenario_state(self):
        with open(self.output_path, 'w') as f:
            for line in self.data:
                f.write(json.dumps(line) + '\n')
            
            
            
def get_request_processor(args, tokenizer):
    if args.infer_data.strip() == '':
        return DummyRequestProcessor(tokenizer)
    else:
        return RequestProcessor(args.infer_data, tokenizer)
