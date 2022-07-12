import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        with open(path) as f:
            self.data = json.load(f)['request_states']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.tokenizer(
            self.data[idx]['request']['prompt'], return_tensors='pt', 
            padding='max_length', truncation=True,
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item['text'] = item['input_ids']

        return item
    

def get_inference_data_loader(args, tokenizer, num_workers=0):
    test_dataset = JsonDataset(args.infer_data, tokenizer)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=None
    )
    print('warning: drop last now, but in practice it should not be dropped.')
    return test_data_loader