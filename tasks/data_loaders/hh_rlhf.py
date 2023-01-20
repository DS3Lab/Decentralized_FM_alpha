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
        self.data = data # Dahoas/full-hh-rlhf ----# Anthropic/hh-rlhf
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        self.rewards = []
        
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
        rewards = self.rewards
        for x in self.data:
            self.iter_count += 1
            # prompt = x['prompt'].replace(' Assistant:', '\n\nAssistant:').replace(' Human:', '\n\User:')
            chosen = x['chosen'].replace('Human:', 'User:')
            rejected = x['rejected'].replace('Human:', 'User:')
            
            # curr_tokens = self.tokenizer(prompt)['input_ids']
            # buffer_tokens += curr_tokens
            # rewards += [1.0] * curr_tokens
            
#             if random.random() <= 1.0:
#                 curr_tokens = self.tokenizer(chosen)['input_ids']
#                 buffer_tokens += curr_tokens
#                 rewards += [1.0] * len(curr_tokens)
                
#             else:
#                 curr_tokens = self.tokenizer(rejected)['input_ids']
#                 buffer_tokens += curr_tokens
#                 rewards += [-0.1] * len(curr_tokens)

            curr_tokens = self.tokenizer(chosen)['input_ids']
            buffer_tokens += curr_tokens
            rewards += [1.0] * len(curr_tokens)
                
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                weights = rewards[:self.seq_length]
                buffer_tokens = buffer_tokens[self.seq_length:]
                rewards = rewards[self.seq_length:]
                input_ids = torch.tensor(tokens)
                weights = torch.tensor(weights)
                self.buffer_tokens = buffer_tokens # update for restore
                self.rewards = rewards
                yield {
                    'input_ids': input_ids,
                    # 'weights': weights,
                }
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
        
    
# def get_pile_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
#     data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
#     stream_dataset = StreamDataset(data, tokenizer, args.seq_length)
    
#     if state_dict is not None:
#         stream_dataset.load_state_dict(state_dict)
    
#     train_data_loader = torch.utils.data.DataLoader(stream_dataset,
#                                                     batch_size=args.batch_size * args.data_group_size,
#                                                     shuffle=False,
#                                                     num_workers=num_workers,
#                                                     pin_memory=True,
#                                                     collate_fn=None)
#     return train_data_loader