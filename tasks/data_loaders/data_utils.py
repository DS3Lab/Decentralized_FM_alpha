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

SHOW_DATA = int(os.environ.get('SHOW_DATA', 1))

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
    
    
class OIGAugmentProcessor:
    
    def __init__(self, tokenizer, seq_length=1024):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
    
        import random
        from augmentations.mild_mix_perturbation import MildMixPerturbation
        self.p = MildMixPerturbation()
        self.rng = random
        
    def __call__(self, inputs):
        
        tokens = inputs['input_ids']
        text = self.tokenizer.decode(tokens)
        
        final_text = ''

        if text.startswith('User:') or text.startswith('Assistant:'):
            text = '\n' + text
        for i, chunk in enumerate(text.split('\nUser:')):
            if i == 0:
                final_text += chunk
                continue
            if '\nAssistant:' in chunk:
                user_chunk, assistant_chunk = chunk.split('\nAssistant:')[:2]
                user_chunk = user_chunk.strip()
                if user_chunk != '':
                    final_text += '\nUser: ' + self.p.perturb(user_chunk, rng=self.rng)
                assistant_chunk = assistant_chunk.strip()
                if assistant_chunk != '':
                    final_text += '\nAssistant: ' + assistant_chunk
            else:
                chunk = chunk.strip()
                final_text += '\nUser:' + chunk
                
        text = final_text
        
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.seq_length]
        tokens = tokens + (self.seq_length - len(tokens)) * [self.tokenizer.eos_token_id]
        
        return {
            'input_ids': torch.tensor(tokens),
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
                    
                    if SHOW_DATA:
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
        print(task)
        if task == 'natural_instructions' or task == 'ni':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset('./natural-instructions/', tokenizer, args.seq_length)
        elif task == 'ni_chat':
            from .natural_instructions_chat import StreamDataset
            dataset = StreamDataset('./natural-instructions/', tokenizer, args.seq_length)
        elif task == 'p3':
            from .p3 import StreamDataset
            data = load_dataset("Muennighoff/P3", split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'flan':
            from .p3 import StreamDataset
            data = load_dataset("Muennighoff/flan", split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed).with_format("torch")
            # data = load_dataset('the_pile', split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'lawinstruct':
            from .pile import StreamDataset
            data_files = {"train": "data/*"}
            data = load_dataset('lawinstruct/lawinstruct', split='train', data_files=data_files, use_auth_token=True, streaming=True).shuffle(buffer_size=1000_000, seed=args.seed).with_format("torch")
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'lawinstruct_en':
            from .pile import StreamDataset
            data_files = {"train": "data/*"}
            data = load_dataset('lawinstruct/lawinstruct', split='train', data_files=data_files, use_auth_token=True, streaming=True)
            data = data.filter(lambda x: x['lang']=='en').shuffle(buffer_size=100_000, seed=args.seed).with_format("torch")
            dataset = StreamDataset(data, tokenizer, args.seq_length, splitter='\n\n\n')
        elif task == 'multi_legal_pile_en':
            from .pile import StreamDataset
            data = load_dataset('joelito/Multi_Legal_Pile', 'en_all', split='train', streaming=True).shuffle(buffer_size=1000_000, seed=args.seed).with_format("torch")
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'multi_legal_pile_filtered_en':
            from .pile import StreamDataset
            data = load_dataset('joelito/MultiLegalPile_Wikipedia_Filtered', 'en_all', split='train', streaming=True)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'c4':
            from .c4 import StreamDataset
            data = load_dataset('c4', 'en', split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed)
            # data = load_dataset('c4', 'en', split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset('./data/mmlu-cot.json', tokenizer, args.seq_length)
        elif task == 'hc3':
            from .hc3 import StreamDataset
            data = load_dataset('Hello-SimpleAI/HC3', 'all', split='train')
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'hh_rlhf':
            from .hh_rlhf import StreamDataset
            data = load_dataset('Anthropic/hh-rlhf', split='train').shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'unatural_instructions':
            from .pile import StreamDataset
            data = load_dataset("json", data_files='./data/unatural_instructions.jsonl', split="train", streaming=True).shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length, doc_separator='\n')
        elif task == 'c4_chat':
            from .pile_chat import StreamDataset
            data = load_dataset('c4', 'en', split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif 'safety' in task:
            from .safety import StreamDataset
            data = load_dataset("json", data_files=task, split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif 'alpaca_data.json' in task:
            from .alpaca import StreamDataset
            dataset = StreamDataset(task, tokenizer, args.seq_length)
        else:
            if 'p3' in task:
                from .p3 import StreamDataset
            elif ('soda' in task) or ('oa_v3_fixed_plus_safety') in task or ('cot_instructions' in task) or ('mix' in task):
                from .pile import StreamDataset
                StreamDataset.default_doc_separator = '\n'
            # if 'jsonl' in task:
            #     from .pile import StreamDataset
            #     StreamDataset.default_doc_separator = '\n'
            else:
                from .pile import StreamDataset
            print('data_utils: before getting custom pile')
            data = load_dataset("json", data_files=task, split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed)
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
    
    # post_processor = OIGAugmentProcessor(tokenizer, seq_length=args.seq_length)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length,
        # post_processor=post_processor,
    )
    
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


def get_imagenet_train_data_loader(args, tokenizer, num_workers=16, state_dict=None):
    
    def process_example(example):
        inputs = tokenizer((example['image'] if example['image'].mode == 'RGB' else example['image'].convert('RGB')), return_tensors='pt')
        inputs['label'] = example['label']
        return inputs
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = tokenizer([(x if x.mode == 'RGB' else x.convert('RGB')) for x in example_batch['image']], return_tensors='pt')
        # Don't forget to include the label!
        inputs['label'] = example_batch['label']
        return inputs
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'label': torch.tensor([x['label'] for x in batch])
        }
    
    # ds = load_dataset('beans', split='train')
    ds = load_dataset(args.task_name, split='train', use_auth_token=True)
    
    ds = ds.select(
        (
            i for i in range(len(ds)) 
            if i not in (25,) # grey image
        )
    )
    
    prepared_ds = ds.with_transform(transform)
    
    train_data_loader = torch.utils.data.DataLoader(prepared_ds,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn)
    
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
