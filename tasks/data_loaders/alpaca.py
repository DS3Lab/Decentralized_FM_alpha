import os
import re
import torch
import json
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *



class StreamDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, seq_length=1024):
        
        self.data_path = data_path
        
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        
        self.iter_count = 0
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
        }
    
    def load_state_dict(self, state_dict):
        try:
            self.iter_count = state_dict['iter_count']
        except:
            print('cannot load ni states.')
        
    def get_sequence(self):
        
        while True:
            
            prompt = ''
            while True:
                item = random.choice(self.data)
                
                if item['output'] == '':
                    continue
                
                if item['input'] != '':
                    prompt += f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}

'''
                else:
                    prompt += f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Response:
{item['output']}

'''   
                
                input_ids = self.tokenizer(prompt.strip())['input_ids']
                if len(input_ids) > self.seq_length:
                    input_ids = input_ids[:self.seq_length]
                    break
                    
            self.iter_count += 1
            
            yield {
                'input_ids': torch.tensor(input_ids),
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
            
        for i in range(self.iter_count):
            next(self.it)
            
        return self.it
    
    
    
def get_natural_instructions_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    stream_dataset = StreamDataset('/root/natural-instructions/', tokenizer, args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader