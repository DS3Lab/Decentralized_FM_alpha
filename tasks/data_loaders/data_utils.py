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


from itertools import islice
from random import randint

def random_chunk(li, min_chunk=1, max_chunk=5):
    it = iter(li)
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break
            

class UL2RProcessor:
    def __init__(self, tokenizer, seq_length=1024):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.s2s_prefix = self.tokenizer("[S2S]")['input_ids']
        self.nlg_prefix = self.tokenizer("[NLG]")['input_ids']
        self.nlu_prefix = self.tokenizer("[NLU]")['input_ids']
        
        self.extra_ids = [self.tokenizer.eos_token_id - 100 + i for i in range(80)]
        
        
    def preprocess_tokens_s2s(self, tokens):
        
        tokens = self.s2s_prefix + tokens
        
        split = int(random.random() * len(tokens))
        
        tokens = tokens[:split] + tokens[split:]
        tokens = tokens[:self.seq_length]
        
        prefix_masks = torch.zeros(len(tokens), dtype=torch.uint8)
        prefix_masks[:split] = 1
        
        return {
            'input_ids': torch.tensor(tokens),
            'prefix_masks': prefix_masks,
        }
    
    def preprocess_tokens_nlg(self, tokens):
        
        tokens = tokens[:self.seq_length - len(self.nlg_prefix) - 2]
        
        start = int(random.random() * len(tokens))
        end = start + 1 + int(random.random() * 31)
        
        left = self.nlg_prefix + tokens[:start] + [self.extra_ids[0]] + tokens[end:]
        right = [self.extra_ids[0]] + tokens[start:end]
    
        tokens = left + right
        tokens = tokens[:self.seq_length]
        tokens = tokens + (self.seq_length - len(tokens)) * [self.tokenizer.eos_token_id]
        
        prefix_masks = torch.zeros(len(tokens), dtype=torch.uint8)
        prefix_masks[:len(left)] = 1
        
        return {
            'input_ids': torch.tensor(tokens),
            'prefix_masks': prefix_masks,
        }
        
    def preprocess_tokens_nlu(self, tokens):
        
        tokens = tokens[:self.seq_length - len(self.nlu_prefix) - 10]
        
        # split to chunks
        chunks = list(random_chunk(tokens, min_chunk=1, max_chunk=5))
        
        # randomly select 15%
        K = int(0.15 * len(chunks))
        indices = random.sample(range(len(chunks)), K)
        
        left = self.nlu_prefix
        right = []
        extra_id_count = 0
        
        last_corrupt = False
        for i, chunk in enumerate(chunks):
            # make sure not consecutive corrupt chunks
            if i in indices and not last_corrupt and extra_id_count < len(self.extra_ids):
                left += [self.extra_ids[extra_id_count]]
                right += [self.extra_ids[extra_id_count]] + chunk
                extra_id_count += 1
            else:
                left += chunk
                last_corrupt = False
        
        tokens = left + right
        tokens = tokens[:self.seq_length]
        tokens = tokens + (self.seq_length - len(tokens)) * [self.tokenizer.eos_token_id]
        
        prefix_masks = torch.zeros(len(tokens), dtype=torch.uint8)
        prefix_masks[:len(left)] = 1
        
        return {
            'input_ids': torch.tensor(tokens),
            'prefix_masks': prefix_masks,
        }

        
    # def __call__(self, inputs):
    #     tokens = inputs['input_ids'].tolist()
    #     p = random.random()
    #     if p > 0.5:
    #         return self.preprocess_tokens_s2s(tokens)
    #     elif p > 0.25:
    #         return self.preprocess_tokens_nlg(tokens)
    #     else:
    #         return self.preprocess_tokens_nlu(tokens)
    
    def __call__(self, inputs):
        
        tokens = inputs['input_ids'].tolist()
        
        if random.random() < 0.2:
            split = int(random.random() * 20)
        else:
            split = int(random.random() * len(tokens))
        
        tokens = tokens[:split] + tokens[split:]
        tokens = tokens[:self.seq_length]
        
        prefix_masks = torch.zeros(len(tokens), dtype=torch.uint8)
        prefix_masks[:split] = 1
        
        return {
            'input_ids': torch.tensor(tokens),
            'prefix_masks': prefix_masks,
        }


class StreamDatasetList(IterableDataset):
    def __init__(self, task_names, datasets, sample_probs, tokenizer, seq_length=1024, print_sample_every_n=64, post_processor=None):
        
        self.task_names = task_names
        self.datasets = datasets
        self.sample_probs = sample_probs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.print_sample_every_n = print_sample_every_n
        self.post_processor = post_processor
        
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
            
            for task_name, it, th in zip(self.task_names, iterators, prob_ths):
                if p < th:
                    
                    inputs = next(it)
                    
                    if self.post_processor is not None:
                        inputs = self.post_processor(inputs)
                    
                    if global_i % self.print_sample_every_n == 0:
                        print(p, th)
                        print(f"**{task_name}**:", self.tokenizer.decode(inputs['input_ids']))
                        
                    yield inputs
                    global_i += 1
                    break
                    
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
def name_to_dataset(task, tokenizer, args):
    
    if task != '':
        if task == 'natural_instructions' or task == 'ni':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset('./natural-instructions/', tokenizer, args.seq_length)
        elif task == 'p3':
            from .p3 import StreamDataset
            data = load_dataset("Muennighoff/P3", split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            print('data_utils: before getting pile')
            data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed).with_format("torch")
            print('data_utils: after getting pile')
            # data = load_dataset('the_pile', split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'c4':
            from .c4 import StreamDataset
            # data = load_dataset('c4', 'en', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
            data = load_dataset('c4', 'en', split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset('./data/mmlu-cot.json', tokenizer, args.seq_length)
        else:
            from .pile import StreamDataset
            print('data_utils: before getting custom pile')
            data = load_dataset("json", data_files=task, split="train", streaming=True).shuffle(seed=args.seed)
            print('data_utils: after getting custom pile')
            dataset = StreamDataset(data, tokenizer, args.seq_length)
            # print('unknow task {task}, skip.')
            # assert False
        
    return dataset

    
def get_train_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    
    print('data_utils: parse task_list')
    
    for task in task_list:
        if ':' in task:
            task, prob = task.strip().split(':')
            prob = float(prob)
        else:
            task = task.strip()
            prob = 1.0
            
        dataset = name_to_dataset(task, tokenizer, args)
            
        print('data_utils:', task, prob)
    
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
    
    print('data_utils: get train_data_loader')
    
    return train_data_loader



def get_ul2r_train_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
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
            
        dataset = name_to_dataset(task, tokenizer, args)
    
        task_names.append(task)
        datasets.append(dataset)
        probs.append(prob)
        
    ul2r_processor = UL2RProcessor(tokenizer, seq_length=args.seq_length)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length, post_processor=ul2r_processor)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    
    print('ul2r dataloader init done.')
    
    return train_data_loader