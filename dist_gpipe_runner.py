import argparse
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
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
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

    if args.profiling == 'no-profiling':
        distributed_train_foo_iter(args, gpipe, device, train_data_loader)
    else:
        trace_file = './trace_json/gpt3_gpipe_b' + str(args.batch_size) + '_' + str(args.micro_batch_size) + \
                     '_l' + str(args.seq_length) + '_m' + str(args.embedding_dim) + '_w' + str(args.world_size) + \
                     '_' + str(args.rank) + '_' + args.profiling + '.json'
        if args.profiling == 'tidy_profiling':
            distributed_train_foo_iter(args, gpipe, device, train_data_loader)
            gpipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                distributed_train_foo_iter(args, gpipe, device, train_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False


if __name__ == '__main__':
    main()
