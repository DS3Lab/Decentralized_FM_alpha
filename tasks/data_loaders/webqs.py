
from ..tasks.webqs import WebQs


class WebQsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def _get_dataset(task, tokenizer, data_split='train', data_style='bert'):
    if data_split == 'train':
        data = task.training_docs()
    elif data_split == 'validation':
        data = task.validation_docs()
    elif data_split == 'test':
        data = task.test_docs()
    else:
        raise Exception('Unrecognized data split.')
        
    questions = []
    answers = []
    for qa in data:
        questions.append(qa['question'])
        answers.append(qa['answers'][0]) # only take the first answer
                
    if tokenizer.pad_token is None:
        print('Setting pad_token to eos_token...')
        tokenizer.pad_token = tokenizer.eos_token
    
    if data_style == 'bert':
    
        encodings = tokenizer(contexts, questions, truncation=True, padding=True)

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        
    elif data_style == 'gpt':
        
        pass
    
    else:
        raise Exception('Unrecognized data style.')
    
    dataset = WebQsDataset(encodings)
    
    return dataset


def get_webqs_train_data_loader(args, tokenizer, num_workers=0):
    task = WebQs()
    train_dataset = _get_dataset(task=task, tokenizer=tokenizer, data_split='train', data_style='bert')
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader