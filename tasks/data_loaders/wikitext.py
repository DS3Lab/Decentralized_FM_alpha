
import torch
from ..tasks.wikitext import WikiText


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def _get_dataset(task, tokenizer, data_split='train', data_style='bert'):
    if data_split == 'train':
        data = task.training_docs()
    elif data_split == 'validation':
        data = task.validation_docs()
    elif data_split == 'test':
        data = task.test_docs()
    else:
        raise Exception('Unrecognized data split.')
        
    if tokenizer.pad_token is None:
        print('Setting pad_token to eos_token...')
        tokenizer.pad_token = tokenizer.eos_token
    
#     encodings = tokenizer(list(data), truncation=True, padding=True)
#     encodings['labels'] = encodings['input_ids']
    encodings = {
        'input_ids': [],
        'labels': [],
#         'attention_mask': [],
    }
    for doc in data:
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(doc) + [tokenizer.eos_token_id]
        input_ids = input_ids[:tokenizer.model_max_length]
        attention_mask = [1]*len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id]*(
            tokenizer.max_len_single_sentence - len(input_ids))
        attention_mask = attention_mask + [0]*(
            tokenizer.max_len_single_sentence - len(attention_mask))
        
        encodings['input_ids'].append(input_ids)
        encodings['labels'].append(input_ids)
#         encodings['attention_mask'].append(attention_mask)
    
    dataset = _Dataset(encodings)
    
    return dataset


def get_wikitext_data_loader(args, tokenizer, data_split='validation', num_workers=0):
    task = WikiText()
    train_dataset = _get_dataset(task=task, tokenizer=tokenizer, data_split=data_split)
    _generator = torch.Generator()
    _generator.manual_seed(42)
    train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=_generator)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader