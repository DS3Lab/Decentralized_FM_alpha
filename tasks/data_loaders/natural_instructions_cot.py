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
    def __init__(self, data_path, cot_data_path, tokenizer, seq_length=1024):
        
        self.data_path = data_path
        self.cot_data_path = cot_data_path
        
        with open(cot_data_path) as f:
            self.cot_data = json.load(f)
        
        self.train_splits = []
        with open(os.path.join(data_path, 'splits/default/train_tasks.txt')) as f:
            for line in f:
                if line.strip() == '':
                    continue
                self.train_splits.append(line.strip() + '.json')
        
        self.task_paths = [
            os.path.join(data_path, 'tasks', p) for p in os.listdir(os.path.join(data_path, 'tasks')) if p.endswith('.json') and p in self.train_splits
        ]
        self.tasks = []
        for task_path in self.task_paths:
            with open(task_path) as f:
                self.tasks.append(json.load(f))
        
        self.buffer_tokens = []
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        
        self.input_prefixs = ['Input: ', 'Given: ', 'Context: ', 'Example: ', 'Question: ', '']
        self.output_prefixs = ['Output: ', 'Ans: ', 'A: ', 'Answer: ', 'Return: ', 'R: ', '']
        self.splitters = ['\n', '\n\n', '\n\n\n', ' ', '\t', '\t\t', '##', '###']
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
    
    def sample_text_from_task(self, task):
        
        text_splitter = random.choice(self.splitters)
        text_def = random.choice(task['Definition'] + [""]).strip()
        text_input = random.choice(self.input_prefixs)
        text_output = random.choice(self.output_prefixs)

        text_context = text_def
        
        while True:
            instance = random.choice(task['Instances'])
            text_context += text_splitter + text_input + instance['input'] + text_splitter + text_output + random.choice(instance['output'])
            input_ids = self.tokenizer(text_context)['input_ids']
            if len(input_ids) > self.seq_length:
                break
                
        input_ids = input_ids[:self.seq_length]
        input_ids = torch.tensor(input_ids).long()
        
        return input_ids
    
    def get_sequence_from_cot(self):
        
        while True:
            
            keys = list(self.cot_data.keys())
            random.shuffle(keys)
            
            for k in keys:
                
                v = self.cot_data[k]
                
                input_ids = self.tokenizer(v)['input_ids']
                if len(input_ids) < self.seq_length:
                    input_ids += [self.tokenizer.eos_token_id]*(self.seq_length - len(input_ids))
                
                input_ids = input_ids[:self.seq_length]
                input_ids = torch.tensor(input_ids).long()
                
                yield input_ids
        
    def get_sequence(self):
        
        it_cot = cycle(self.get_sequence_from_cot())
        
        while True:
            
            if random.random() < 0.9:
            
                task = random.choice(self.tasks)
                input_ids = self.sample_text_from_task(task)
                
            else:
                
                input_ids = next(it_cot)
                

            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
    
def get_natural_instructions_cot_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    stream_dataset = StreamDataset(
        '/root/natural-instructions/',
        './data/mmlu-cot.json',
        tokenizer=tokenizer, seq_length=args.seq_length
    )
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader