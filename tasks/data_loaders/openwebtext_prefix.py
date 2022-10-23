import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


class StreamDataset(IterableDataset):
    def __init__(self, data, tokenizer, seq_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = [self.tokenizer.bos_token_id]
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
            'buffer_tokens': self.buffer_tokens,
        }
    
    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.buffer_tokens = state_dict['buffer_tokens']
        self.data = self.data.skip(self.iter_count)
        
    def preprocess_tokens(self, tokens):
        
#         p = random.random()
#         if p > 0.5:
#             mode = "[S2S]"
#         elif p > 0.25:
#             mode = "[NLG]"
#         else:
#             mode = "[NLU]"
        
#         if mode == "[S2S]":
        
        split = int(random.random() * len(tokens))
        
        tokens = tokens[:split] + [self.tokenizer.bos_token_id] + tokens[split:]
        tokens = tokens[:self.seq_length]
        
        prefix_masks = torch.zeros(len(tokens), dtype=torch.uint8)
        prefix_masks[:split] = 1
        
        return tokens, prefix_masks
        
    def get_sequence(self):
        buffer_tokens = self.buffer_tokens
        for x in self.data:
            self.iter_count += 1
            curr_tokens = self.tokenizer(x['text'])['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length - 1:
                tokens = buffer_tokens[:self.seq_length-1]
                buffer_tokens = [self.tokenizer.bos_token_id] + buffer_tokens[self.seq_length-1:]
                tokens, prefix_masks = self.preprocess_tokens(tokens)
                input_ids = torch.tensor(tokens)
                self.buffer_tokens = buffer_tokens # update for restore
                yield {
                    'input_ids': input_ids,
                    'prefix_masks': prefix_masks,
                }
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
def get_openwebtext_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    data = load_dataset('openwebtext', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
    stream_dataset = StreamDataset(data, tokenizer, args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader
    