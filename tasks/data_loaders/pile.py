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
    default_doc_separator = ''
    def __init__(self, data, tokenizer, seq_length=1024, doc_separator=None):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.doc_separator = doc_separator or StreamDataset.default_doc_separator
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        
        self.greetings = [
         "<human>: Hello\n<bot>: Hello! How may I help you today?",
         "<human>: Good morning\n<bot>: Good morning! How may I help you today?",
         "<human>: Good afternoon\n<bot>: Good afternoon! How may I help you today?",
         "<human>: Good evening\n<bot>: Good evening! How may I help you today?",
         "<human>: How are you?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: How are you doing?\n<bot>: I'm doing well, thank you! How may I help you today?",
         "<human>: Nice to meet you\n<bot>: Nice to meet you too! How may I help you today?",
         "<human>: It's nice to meet you\n<bot>: Nice to meet you too! How may I help you today?",
         "<human>: I'm pleased to meet you.\n<bot>: Me too! How may I help you today?",
         "<human>: It's a pleasure to meet you.\n<bot>: Me too! How may I help you today?",
         "<human>: I'm glad to see you.\n<bot>: Glad to meet you too! How may I help you today?",
         "<human>: How do you do?\n<bot>: Hi! How may I help you today?",
         "<human>: Hi\n<bot>: Hi! How may I help you today?",
         "<human>: Hey\n<bot>: Hi! How may I help you today?",
         "<human>: What's up?\n<bot>: Hi! How may I help you today?",
         "<human>: How's it going?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: How have you been?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: What's new?\n<bot>: Hi! How may I help you today?",
         "<human>: What's going on?\n<bot>: Hi! How may I help you today?",
         "<human>: How are things?\n<bot>: Hi! How may I help you today?",
         "<human>: How's your day?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: How's your day going?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: Good to see you.\n<bot>: Hi! How may I help you today?",
         "<human>: Long time no see.\n<bot>: Hi! How may I help you today?",
         "<human>: It's been a while.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: It's been a long time.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: It's been such a long time.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: It's been too long.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: I'm so happy to see you again.\n<bot>: Me too! How may I help you today?",
         "<human>: Wow, it's so good to see you again!\n<bot>: Me too! How may I help you today?",
         "<human>: What have you been up to?\n<bot>: Hi! How may I help you today?",

         "<human>: hello\n<bot>: Hello! How may I help you today?",
         "<human>: good morning\n<bot>: Good morning! How may I help you today?",
         "<human>: good afternoon\n<bot>: Good afternoon! How may I help you today?",
         "<human>: good evening\n<bot>: Good evening! How may I help you today?",
         "<human>: how are you?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: how are you doing?\n<bot>: I'm doing well, thank you! How may I help you today?",
         "<human>: nice to meet you\n<bot>: Nice to meet you too! How may I help you today?",
         "<human>: it's nice to meet you\n<bot>: Nice to meet you too! How may I help you today?",
         "<human>: i'm pleased to meet you.\n<bot>: Me too! How may I help you today?",
         "<human>: it's a pleasure to meet you.\n<bot>: Me too! How may I help you today?",
         "<human>: i'm glad to see you.\n<bot>: Glad to meet you too! How may I help you today?",
         "<human>: how do you do?\n<bot>: Hi! How may I help you today?",
         "<human>: hi\n<bot>: Hi! How may I help you today?",
         "<human>: hey\n<bot>: Hi! How may I help you today?",
         "<human>: what's up?\n<bot>: Hi! How may I help you today?",
         "<human>: how's it going?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: how have you been?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: what's new?\n<bot>: Hi! How may I help you today?",
         "<human>: what's going on?\n<bot>: Hi! How may I help you today?",
         "<human>: how are things?\n<bot>: Hi! How may I help you today?",
         "<human>: how's your day?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: how's your day going?\n<bot>: Great, thank you! How may I help you today?",
         "<human>: good to see you.\n<bot>: Hi! How may I help you today?",
         "<human>: long time no see.\n<bot>: Hi! How may I help you today?",
         "<human>: it's been a while.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: it's been a long time.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: it's been such a long time.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: it's been too long.\n<bot>: Yes, it has! How may I help you today?",
         "<human>: i'm so happy to see you again.\n<bot>: Me too! How may I help you today?",
         "<human>: wow, it's so good to see you again!\n<bot>: Me too! How may I help you today?",
         "<human>: what have you been up to?\n<bot>: Hi! How may I help you today?",
        ]
        
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
        while True:
            try:
                for x in self.data.shuffle():
                    self.iter_count += 1
                    curr_tokens = self.tokenizer(self.doc_separator + x['text'])['input_ids']
                    buffer_tokens += curr_tokens
                    while len(buffer_tokens) >= self.seq_length:
                        tokens = buffer_tokens[:self.seq_length]
                        buffer_tokens = buffer_tokens[self.seq_length:]
                        input_ids = torch.tensor(tokens)
                        self.buffer_tokens = buffer_tokens # update for restore
                        yield {
                            'input_ids': input_ids,
                        }
            except:
                print('next epoch')
            # break
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
        
    
def get_pile_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    data = load_dataset('the_pile', split="train", streaming=True).shuffle(buffer_size=10_000, seed=args.seed)
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
