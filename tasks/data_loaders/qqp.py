
import os
import torch
from datasets import load_dataset, load_from_disk

# from ..tasks.glue import QQP

# class _Dataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

#     def __len__(self):
#         return len(self.encodings['input_ids'])


# def _get_dataset(task, tokenizer, data_split='train', data_style='bert'):
#     if data_split == 'train':
#         data = task.training_docs()
#     elif data_split == 'validation':
#         data = task.validation_docs()
#     elif data_split == 'test':
#         data = task.test_docs()
#     else:
#         raise Exception('Unrecognized data split.')
        
#     if tokenizer.pad_token is None:
#         print('Setting pad_token to eos_token...')
#         tokenizer.pad_token = tokenizer.eos_token
    
# #     encodings = tokenizer(list(data), truncation=True, padding=True)
# #     encodings['labels'] = encodings['input_ids']
#     encodings = {
#         'input_ids': [],
#         'labels': [],
#         'attention_mask': [],
#     }
#     for doc in data:
#         input_ids = [tokenizer.bos_token_id] + tokenizer.encode(doc['question1'], doc['question2']) + [tokenizer.eos_token_id]
#         input_ids = input_ids[:tokenizer.model_max_length]
#         attention_mask = [1]*len(input_ids)
#         input_ids = input_ids + [tokenizer.pad_token_id]*(
#             tokenizer.max_len_single_sentence - len(input_ids))
#         attention_mask = attention_mask + [0]*(
#             tokenizer.max_len_single_sentence - len(attention_mask))
        
#         encodings['input_ids'].append(input_ids)
#         encodings['labels'].append(input_ids)
#         encodings['attention_mask'].append(attention_mask)
    
#     # set alias
#     encodings['text'] = encodings['input_ids'] 
#     encodings['label'] = encodings['labels'] 
    
#     dataset = _Dataset(encodings)
    
#     return dataset


def get_qqp_data_loader(args, tokenizer, data_split='train', num_workers=0):
    
    
    def _encode(examples):
        return tokenizer(examples['question1'], examples['question2'], 
                         truncation=True, padding='max_length', max_length=args.seq_length)
    
    if os.path.isdir('./data/glue_qqp'):
        train_set = load_from_disk('./data/glue_qqp')[data_split]
    else:
        train_set = load_dataset('glue', 'qqp', split=data_split)
    train_set = train_set.map(_encode, batched=True)
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    if 'token_type_ids' in train_set.features:
        train_set.set_format(
            type='torch', columns=[
                'text', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'idx',
            ])
    else:
        train_set.set_format(
            type='torch', columns=[
                'text', 'input_ids', 'attention_mask', 'label', 'idx',
            ])
    
    
    if data_split == 'train':
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        sampler=train_sampler,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        collate_fn=None)
    else:
        # test or valid data loader
        # TODO: let drop_last be False
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        collate_fn=None)
    return train_data_loader