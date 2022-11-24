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
    def __init__(self, data_path, distill_data_path, cot_data_path, tokenizer, seq_length=1024):
        
        self.data_path = data_path
        self.distill_data_path = distill_data_path
        self.cot_data_path = cot_data_path
        
        with open(cot_data_path) as f:
            self.cot_data = json.load(f)
        
        self.distill_data_paths = [os.path.join(distill_data_path, splitname) for splitname in os.listdir(distill_data_path) if splitname.endswith('.jsonl')]
        self.distill_data = []
        for path in self.distill_data_paths:
            with open(path) as f:
                for line in f:
                    if line.strip() == '':
                        continue
                    item = json.loads(line)
                    self.distill_data.append(item['request']['prompt'] + item['result']['choices'][0]['text'])
        
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
        
        self.input_prefixs = ['Input: ', '', '']
        self.output_prefixs = ['Output: ', 'Ans: ', 'A: ', 'Answer: ', '', '', '']
        self.splitters = ['\n', '\n\n', '\n\n\n', ' ', '  ', '\t', '\t\t', '##', '###']
        
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
    
    def get_sequence_from_distill_data(self):
        
        while True:
            
            random.shuffle(self.distill_data)
            
            for text in self.distill_data:
                
                input_ids = self.tokenizer(text)['input_ids']
                if len(input_ids) < self.seq_length:
                    input_ids += [self.tokenizer.eos_token_id]*(self.seq_length - len(input_ids))
                
                input_ids = input_ids[:self.seq_length]
                input_ids = torch.tensor(input_ids).long()
                
                yield input_ids
                
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
        
        it_distill = cycle(self.get_sequence_from_distill_data())
        it_cot = cycle(self.get_sequence_from_cot())
        
        while True:
            
            p = random.random()
            
            if p < 0.5:
            
                task = random.choice(self.tasks)
                input_ids = self.sample_text_from_task(task)
                
            elif p < 0.9:
                
                input_ids = next(it_distill)
                
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
    
    
    
def get_natural_instructions_distill_cot_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    stream_dataset = StreamDataset(
        '/root/natural-instructions/',
        './data/ni_opt66b/',
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