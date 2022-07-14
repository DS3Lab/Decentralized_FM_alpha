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
                self.data = self.data + [{'request':{'prompt': tokenizer.bos_token}}]*n_dummy
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.tokenizer(
            self.data[idx]['request']['prompt'], return_tensors='pt', 
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
        pass
    
    def get_dataloader(self, batch_size, num_workers=0):
        dataset = JsonDataset(
            [{'request':{'prompt': 'you are'}}]*2000, 
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
        
    def add_result(self, inputs, outputs):
        batch_size = len(inputs['idx'])
        tokenizer = self.tokenizer
        for i in range(batch_size):
            idx = inputs['idx'][i]
            if idx < 0:
                continue
                
            echo_text = tokenizer.decode(
                inputs['text'][i]
            ).replace('<|endoftext|>', '')
            item = [
                {
                    "text": (echo_text + tokenizer.decode(output_ids[i]).replace('<|endoftext|>', '')).strip(),
                    "top_k_per_token": [], # TODO
                } for output_ids in outputs
            ]
            print(item[0])
        
    def write_scenario_state(self):
        pass
    

class RequestProcessor:
    def __init__(self, root, tokenizer):
        self.root = root
        self.tokenizer = tokenizer
        self.request_path = os.path.join(root, 'scenario_state.json')
        self.output_path = os.path.join(root, 'scenario_state_output.json')
        with open(self.request_path) as f:
            self.scenario_state = json.load(f)
        first_request = self.scenario_state['request_states'][0]['request']
        self.top_k = first_request.get('top_k', 1)
        self.top_p = first_request.get('top_p', 1)
        self.temperature = first_request.get('temperature', 1)
        self.echo_prompt = first_request.get('echo_prompt', 1)
        self.top_k_per_token = first_request.get('top_k_per_token', 1)
        self.num_completions = first_request.get('num_completions', 1)
        self.max_tokens = first_request.get('max_tokens', 1)
        
    def set_arguments(self, args):
        args.top_k = self.top_k
        args.top_p = self.top_p
        args.temperature = self.temperature
        args.echo_prompt = self.echo_prompt
        args.top_k_per_token = self.top_k_per_token
        args.num_completions = self.num_completions
        # args.generate_seq_length = self.max_tokens
        
    def get_dataloader(self, batch_size, num_workers=0):
        dataset = JsonDataset(
            self.scenario_state['request_states'], 
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
    
    def add_result(self, inputs, outputs):
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
            item = [
                {
                    "text": (echo_text + tokenizer.decode(output_ids[i]).replace('<|endoftext|>', '')).strip(),
                    "top_k_per_token": [], # TODO
                } for output_ids in outputs
            ]
            self.scenario_state['request_states'][idx]['result'] = item
            print(item[0])
        
    def write_scenario_state(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.scenario_state, f)
            
            
            
def get_request_processor(args, tokenizer):
    if args.infer_data.strip() == '':
        return DummyRequestProcessor(tokenizer)
    else:
        return RequestProcessor(args.infer_data, tokenizer)