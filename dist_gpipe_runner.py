import argparse
import torch
import torch.autograd.profiler as profiler
from dist_gpt_utils import create_dataset, get_batch
from dist_gpipe_module_sync import GpipeSync
from dist_gpipe_module_async import GpipeAsync


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-num', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                         help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--timing', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='event enables timing or not')
    parser.add_argument('--async-mode', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use async mode or not')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    assert (torch.cuda.is_available())
    device = torch.device('cuda', args.cuda_id)

    if args.rank == 0 or args.rank == args.world_size-1:
        train_data, ntokens = create_dataset(args)
    else:
        train_data = None
        ntokens = -1
    iter_num = 5
    if args.async_mode:
        gpipe = GpipeAsync(args, ntokens, device)
    else:
        gpipe = GpipeSync(args, ntokens, device)
    with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
        for i, indices in enumerate(range(0, iter_num*args.seq_length, args.seq_length)):
            data, targets = get_batch(args, indices, train_data)
            print("<========================================>")
            if args.rank == 0:
                print("Handle Batch ", i, data.shape)
            elif args.rank == args.world_size-1:
                print("Handle Batch ", i, targets.shape)
            else:
                print("Handle Batch ", i)
            gpipe.sgd_iter(data, targets)
    print(prof.key_averages().table())
    trace_file = "../trace_json/gpt3_gpipe_b" + str(args.batch_size) + "_l" + str(args.seq_length) + '_m' + \
                 str(args.embedding_size) + "_w" + str(args.world_size) + "_" + str(args.rank) + ".json"
    prof.export_chrome_trace(trace_file)


if __name__ == '__main__':
    main()
