def add_device_arguments(parser):
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--cuda-num', type=int, default=1, metavar='N',
                        help='number of GPUs, if the instance has multiple GPUs.')
    parser.add_argument('--debug-mem', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, we will print some memory stats.')


def add_torch_distributed_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 4)')
    parser.add_argument('--pipeline-group-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--data-group-size', type=int, default=1, metavar='D',
                        help='world-size (default: 1)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')


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
    parser.add_argument('--num-layers', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')


def add_training_hyper_parameter_arguments(parser):
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-size', type=int, default=4, metavar='N',
                        help='input micro batch size for training (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    parser.add_argument('--num-iters', type=int, default=10, metavar='N',
                        help='-')


def get_model_arguments_str(args):
    return '_l' + str(args.seq_length) + '_m' + str(args.embedding_dim)


def get_dist_arguments_str(args, add_rank=True):
    dist_str = '_w' + str(args.world_size)  + '_p' + str(args.pipeline_group_size) + '_d' + str(args.data_group_size)
    if add_rank:
        dist_str = dist_str + '_' + str(args.rank)
    return dist_str


def get_learning_arguments_str(args):
    return '_b' + str(args.batch_size) + '_' + str(args.micro_batch_size)