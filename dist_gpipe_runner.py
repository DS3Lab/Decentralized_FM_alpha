import argparse
import torch
import torch.autograd.profiler as profiler
from glue_dataset.qqp import get_glue_qqp_train_data_loader
from glue_dataset.tokenizer import build_tokenizer
from dist_gpipe_module_sync import GpipeSync
from dist_gpipe_module_async import GpipeAsync
from dist_gpt_utils import *


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--timing', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='event enables timing or not')
    parser.add_argument('--async-mode', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use async mode or not')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    if args.rank == 0 or args.rank == args.world_size-1:
        tokenizer = build_tokenizer(args)
        print("token vocab size:", tokenizer.vocab_size)
        train_data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
        num_classes = 2
        vocab_size = tokenizer.vocab_size
    else:
        train_data_loader = None
        num_classes = 2
        vocab_size = -1

    if args.async_mode:
        gpipe = GpipeAsync(args, vocab_size, num_classes, device)
    else:
        gpipe = GpipeSync(args, vocab_size, num_classes, device) # TODO: this part is not refactored yet.
    with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
        i = 0

    distributed_train_foo_iter(args, gpipe, device, train_data_loader)
    print(prof.key_averages().table())
    trace_file = "../trace_json/gpt3_gpipe_b" + str(args.batch_size) + "_l" + str(args.seq_length) + '_m' + \
                 str(args.embedding_size) + "_w" + str(args.world_size) + "_" + str(args.rank) + ".json"
    prof.export_chrome_trace(trace_file)


if __name__ == '__main__':
    main()
