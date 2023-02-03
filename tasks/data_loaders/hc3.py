import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk

from nltk.tokenize.treebank import TreebankWordDetokenizer

from comm.comm_utils import *


class StreamDataset(IterableDataset):
    def __init__(self, data, tokenizer, seq_length=1024):
        self.data = data
        self.sources = list(set(data['source']))
        self.detokenizer = TreebankWordDetokenizer()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
            'buffer_tokens': self.buffer_tokens,
        }
    
    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.buffer_tokens = state_dict['buffer_tokens']
        self.data = self.data.skip(self.iter_count)
        
    def get_sequence(self):
        buffer_tokens = self.buffer_tokens
        
        it_list = [
            cycle(iter(self.data.filter(lambda x: x['source'] == source).shuffle())) for source in self.sources
        ]
        
        while True:
            self.iter_count += 1
            it = random.choice(it_list)
            text_list = []
            
            while True:
                x = next(it)
                q = self.detokenizer.detokenize(x['question'].strip().split(' '))
                a = self.detokenizer.detokenize(random.choice(x['human_answers']).strip().split(' '))
                text = f"User: {q}\nAssistant: {a}"
                text_list.append(text)
                
                text = '\n'.join(text_list)
                tokens = self.tokenizer(text)['input_ids']
                
                if len(tokens) >= self.seq_length:
                    tokens = tokens[:self.seq_length]
                    input_ids = torch.tensor(tokens)
                    yield {
                        'input_ids': input_ids,
                    }
                    break
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
