
import torch
from datasets import load_dataset

def get_mrpc_data_loader(args, tokenizer, data_split='train', num_workers=0):
    
    
    def _encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], 
                         truncation=True, padding='max_length', max_length=args.seq_length)
    
    train_set = load_dataset('glue', 'mrpc', split='train')
    train_set = train_set.map(_encode, batched=True)
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'label', 'idx',
        ])
    
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
    return train_data_loader