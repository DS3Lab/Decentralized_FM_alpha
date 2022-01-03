from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nccl_backend import *


def init_comm(args):
    if args.dist_backend == 'cupy_nccl':
        comm = NCCLCommunicator(rank=args.rank, intra_gpu_rank=args.cuda_id,
                                world_size=args.world_size, master_ip=args.dist_url)
    else:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                rank=args.rank, world_size=args.world_size)
        comm = dist
    dist.barrier()
    return comm


def get_batch(args, i, data=None):
    if args.rank == 0:
        seq_len = min(args.seq_length, len(data) - 1 - i)
        return data[i:i+seq_len].t(), None
    elif args.rank == args.world_size - 1:
        seq_len = min(args.seq_length, len(data) - 1 - i)
        return None, data[i+1:i+1+seq_len].view(-1)
    else:
        return None, None


def create_dataset(args):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    train_iter, _, _ = WikiText2()
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in train_iter]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    ntokens = len(vocab)
    # Divide the dataset into bsz parts.
    num_batch = data.size(0) // args.batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, num_batch * args.batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(args.batch_size, -1).t().contiguous()
    return data, ntokens