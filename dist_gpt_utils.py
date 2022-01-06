from nccl_backend import *


def add_distributed_arguments(parser):
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


def add_task_arguments(parser):
    parser.add_argument('--train-data', nargs='+', default=['./glue_dataset/data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./glue_dataset/data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--tokenizer-type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-file', type=str, default='./glue_dataset/data/bert-large-cased-vocab.txt', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0, metavar='N',
                        help='-')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128, metavar='N',
                        help='-')


def add_model_arguments(parser):
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')


def add_training_hyper_parameter_arguments(parser):
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-num', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    parser.add_argument('--num_iters', type=int, default=5, metavar='N',
                        help='-')


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


def distributed_train_foo_iter(args, gpipe, device, train_data_loader):
    if args.rank == 0:
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            gpipe.sgd_iter(input_ids, None)
            if i >= args.num_iter:
                break
    elif args.rank == args.world_size - 1:
        for i, data in enumerate(train_data_loader):
            labels = data['label'].to(device)
            gpipe.sgd_iter(None, labels)
            if i >= args.num_iter:
                break
    else:
        i = 0
        while True:
            gpipe.sgd_iter(None, None)
            i += 1
            if i >= args.num_iter:
                break
