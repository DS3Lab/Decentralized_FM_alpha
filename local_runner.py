import random
import numpy as np
import torch
import argparse
from tasks.data_loaders.qqp import get_qqp_data_loader
from tasks.data_loaders.mrpc import get_mrpc_data_loader
from modules.tokenizer import build_tokenizer
from modules.gpt_modules import GPTForClassification, GPTConfig
from modules.gpt_modules import *
from transformers import GPT2ForSequenceClassification as GPTForClassification

import wandb


def main():
    parser = argparse.ArgumentParser(description='Test pretrain dataset')
    parser.add_argument('--model-name', type=str, default=None, metavar='S',
                        help='which model checkpoint to use. e.g. gpt2.')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=128, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=12, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=12, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='-')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='-')
    args = parser.parse_args()
    
    wandb.init(project='test', entity='pipeline-activation-compression')
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    tokenizer = build_tokenizer(args)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("token vocab size:", tokenizer.vocab_size)
    data_loader = get_mrpc_data_loader(args, tokenizer)
    
    if args.model_name:
        model = GPTForClassification.from_pretrained(args.model_name)
        tokenizer.model_max_length = args.seq_length
    else:
        config = GPTConfig.from_pretrained('gpt2')
        config.n_embd = args.embedding_dim
        config.n_inner = args.embedding_dim*4
        config.n_layer = args.num_layers
        config.n_head = args.num_heads
        config.n_positions = args.seq_length
        config.n_ctx = args.seq_length
        config.vocab_size = tokenizer.vocab_size
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        tokenizer.model_max_length = args.seq_length
        model = GPTForClassification(config=config)
    model = model.to(device)
    model.train()
    model.config.pad_token_id = model.config.eos_token_id
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for e in range(10):
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            outputs = model(input_ids=data['input_ids'], labels=data['label'])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            wandb.log({'loss': loss.item()})
            optimizer.step()
            print("Iter ", i, "===== Loss: ", loss.item(), "======")


if __name__ == '__main__':
    main()
