import torch
import argparse
from tasks.data_loaders.wikitext import get_wikitext_data_loader
from modules.tokenizer import build_tokenizer
from modules.gpt_modules import GPTLMHeadModel, GPTConfig


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
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=1024, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='-')
    args = parser.parse_args()
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    tokenizer = build_tokenizer(args)
    
    print("token vocab size:", tokenizer.vocab_size)
    data_loader = get_wikitext_data_loader(args, tokenizer)
    if args.model_name:
        model = GPTLMHeadModel.from_pretrained(args.model_name)
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
        model = GPTLMHeadModel(config=config)
    model = model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for i, data in enumerate(data_loader):
        
        data = {k: v.to(device) for k, v in data.items()}
        
        optimizer.zero_grad()
        outputs = model(**data)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print("Iter ", i, "===== Loss: ", loss.item(), "======")


if __name__ == '__main__':
    main()
