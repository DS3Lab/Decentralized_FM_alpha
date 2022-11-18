import os
import re
import torch
import json
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *



class StreamDatasetList(IterableDataset):
    def __init__(self, task_names, datasets, sample_probs, tokenizer, seq_length=1024, print_sample_every_n=64):
        
        self.task_names = task_names
        self.datasets = datasets
        self.sample_probs = sample_probs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.print_sample_every_n = print_sample_every_n
        
        self.it = None
        
    def state_dict(self):
        return {
            t: d.state_dict() for t, d in zip(self.task_names, self.datasets)
        }
    
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self.datasets[self.task_names.index(k)].load_state_dict(v)
        
    def get_sequence(self):
        
        iterators = [cycle(d.get_sequence()) for d in self.datasets]
        prob_ths = np.cumsum([p / sum(self.sample_probs) for p in self.sample_probs])
        
        print('prob thresholds:', prob_ths)
        
        global_i = 0
        
        while True:
            
            p = random.random()
            
            
            for it, th in zip(iterators, prob_ths):
                if p < th:
                    inputs = next(it)
                    
                    if global_i % self.print_sample_every_n == 0:
                        print(self.tokenizer.decode(inputs['input_ids']))

                    yield inputs
                    global_i += 1
                    
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
    
def get_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    for task in task_list:
        if ':' in task:
            task, prob = task.strip().split(':')
            prob = float(prob)
        else:
            task = task.strip()
            prob = 1.0
            
        if task == 'natural_instructions':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset('/root/natural-instructions/', tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
            stream_dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'c4':
            from .c4 import StreamDataset
            data = load_dataset('c4', 'en', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
            stream_dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset('./data/mmlu-cot.json', tokenizer, args.seq_length)
        else:
            print('unknow task {task}, skip.')
    
        task_names.append(task)
        datasets.append(dataset)
        probs.append(prob)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader