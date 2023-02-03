import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


from itertools import islice
from random import randint

def random_chunk(li, min_chunk=32, max_chunk=256):
    it = iter(li)
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break


class StreamDataset(IterableDataset):
    default_doc_separator = '\n'
    def __init__(self, data, tokenizer, seq_length=1024, doc_separator=None):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.doc_separator = doc_separator or StreamDataset.default_doc_separator
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        
        self.cmds_complete = [
            'Please complete the above passage',
            "Kindly finish the above text.",
            "Could you please complete the passage above?",
            "Please fill in the rest of the text above.",
            "Please add to the above passage.",
            "Could you please provide the missing information in the above passage?",
            "Please finish writing the above paragraph.",
            "Please continue the text above",
            "Please complete the above section.",
            "Please add the missing content to the passage above",
            "Complete the above passage.",
            "Finish the text above",
            "Fill in the rest of the text above",
            "Add to the above passage.",
            "Provide the missing information in the above passage",
            "Finish writing the above paragraph.",
            "Continue the text above.",
            "Complete the above section.",
            "Add the missing content to the passage above",
        ]
        
        self.cmds_continue = [
            "continue",
            "Continue",
            "continue",
            "Go on",
            "Next",
            "next",
            "move on",
            "what's next?",
            "what's next",
            "go ahead",
        ]
        
        self.cmds_switch = [
            "let's change a topic.",
            "Shall we talk about something else?",
            "How about we switch to a different subject?",
            "I think it's time to move on to another topic.",
            "Let's shift gears and discuss something else.",
            "I'm getting tired of this conversation, let's change it up.",
            "Let's change the subject, what do you say?",
            "Can we please talk about something else now?",
            "I think it's time to change the topic of discussion.",
            "Let's switch to a different conversation.",
            "Let's talk about something else for a change.",
        ]
        
        self.ans_switch = [
            "Sure, what's on your mind?",
            "OK, what would you like to discuss?",
            "Alright, what's the new topic?",
            "Absolutely, what's next do you want to talk about?",
            "Yes, what do you have in mind?",
            "Alright, what's on tap for conversation?",
            "Of course, what would you like to talk about now?",
            "Sure thing, what's the new subject?",
            "Absolutely, what's the next topic of discussion?",
            "Sure, what else would you like to talk about?",
        ]
        
        self.cmds_read = [
            "Read the following passage:",
            "Please take a look at the following text:"
            "Kindly go through the following passage:"
            "Please read the following excerpt:"
            "Would you mind reading the following passage?"
            "Can you take a moment to read the following?"
            "I would like you to read the following:"
            "Could you please have a look at the following text?"
            "Take a moment to read the following:"
            "Kindly peruse the following passage:"
            "Please give the following a read:"
        ]
        
        self.splitters  = ['\n', '\n\n']
        
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
        for x in self.data:
            self.iter_count += 1
            curr_tokens = self.tokenizer(x['text'])['input_ids']
            
            if len(curr_tokens) < 256:
                continue
                
            # print(x['text'])
            
            token_chunks = list(random_chunk(curr_tokens, min_chunk=32, max_chunk=128))
            
            if len(token_chunks) > 3:
                truncated = True
                token_chunks = token_chunks[:3]
            else:
                truncated = False
            
            text = '\nUser: ' + random.choice(self.cmds_read) + ' '
            text += self.tokenizer.decode(token_chunks[0]).strip()
            text += '\n' + random.choice(self.cmds_complete)
            text += '\nAssistant: ' + self.tokenizer.decode(token_chunks[1]).strip()
            
            for chunk in token_chunks[2:]:
                text += '\nUser: ' + random.choice(self.cmds_continue)
                text += '\nAssistant: ' + self.tokenizer.decode(chunk).strip()
                
            if truncated:
                text += '\nUser: ' + random.choice(self.cmds_switch)
                text += '\nAssistant: ' + random.choice(self.ans_switch)
            
            curr_tokens = self.tokenizer(text)['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                buffer_tokens = buffer_tokens[self.seq_length:]
                input_ids = torch.tensor(tokens)
                self.buffer_tokens = buffer_tokens # update for restore
                yield {
                    'input_ids': input_ids,
                }
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
        