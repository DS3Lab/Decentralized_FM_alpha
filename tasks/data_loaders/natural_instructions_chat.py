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
    def __init__(self, data_path, tokenizer, seq_length=1024):
        
        self.data_path = data_path
        
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
        self.classification_tasks = []
        for task_path in self.task_paths:
            with open(task_path) as f:
                task = json.load(f)
                
            output_space = set()
            is_classification = True
            for instance in task['Instances']:
                output_space.add(instance['output'][0])
                if len(output_space) > 10:
                    is_classification = False
                    break
            task['IsClassification'] = is_classification
            task['OutputSpace'] = sorted(list(output_space)) if is_classification else None
            if is_classification:
                self.classification_tasks.append(task)
            self.tasks.append(task)
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        
        self.input_prefixs = ['<human>: ']
        self.output_prefixs = ['<bot>: ', '<bot>: ', '<bot>: ', '<bot>: Answer: ', '<bot>: Label: ',  '<bot>: Output: ']
        self.sample_splitters = ['\n',]
        self.answer_splitters = ['\n',]
        
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
        
        self.iter_count = 0
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
        }
    
    def load_state_dict(self, state_dict):
        try:
            self.iter_count = state_dict['iter_count']
        except:
            print('cannot load ni states.')
    
    def sample_text_from_task(self, task):
        
        '''
        Task Definition(*33%)
        + Output Space(*50%)
        [
            + sample splitter
            + input prefix
            + input
            + answer splitter
            + output prefix
            + output
        ]
        '''
        
        is_classification = task['IsClassification']
        output_space = task['OutputSpace']
        
        sample_splitter = random.choice(self.sample_splitters)
        answer_splitter = random.choice(self.answer_splitters)
        text_def = random.choice(task['Definition']).strip()
        if is_classification and random.random() < 0.5:
            text_def += '\nPossible labels:'
            for i, possible_output in enumerate(output_space):
                text_def += f'\n{i+1}. {possible_output}'
            text_def += '\n'
            
        # text_def = f"<human>: {text_def}\n<bot>: Sure, I understand."
        if random.random() < 0.1:
            greeting = random.choice(self.greetings)
            text_def = f"{greeting}\n<human>: {text_def}"
        else:
            # text_def = f"<human>: {text_def}\n<bot>: Sure, I understand."
            text_def = f"<human>: {text_def}"
        
        text_input = random.choice(self.input_prefixs)
        text_output = random.choice(self.output_prefixs)

        text_context = text_def
        
        while True:
            instance = random.choice(task['Instances'])
            text_context += sample_splitter + text_input + instance['input'] + answer_splitter + text_output + random.choice(instance['output'])
            input_ids = self.tokenizer(text_context.strip())['input_ids']
            if len(input_ids) > self.seq_length:
                break
                
        input_ids = input_ids[:self.seq_length]
        input_ids = torch.tensor(input_ids).long()
        
        return input_ids
        
    def get_sequence(self):
        
        while True:
            
            # ensure at least 30% classification
            if random.random() < 0.3:
                task = random.choice(self.classification_tasks)
            else:
                task = random.choice(self.tasks)

            input_ids = self.sample_text_from_task(task)

            self.iter_count += 1
            
            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
            
        for i in range(self.iter_count):
            next(self.it)
            
        return self.it
    
    
    
def get_natural_instructions_train_data_loader(args, tokenizer, num_workers=0, state_dict=None):
    
    stream_dataset = StreamDataset('/root/natural-instructions/', tokenizer, args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader