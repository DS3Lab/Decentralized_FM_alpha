import argparse
import torch
import torch.autograd.profiler as profiler
from glue_dataset.qqp import get_glue_qqp_train_data_loader
from glue_dataset.tokenizer import build_tokenizer
from pipeline_parallel.dist_1f1b_pipeline_async import Pipe1F1BAsync
from pipeline_parallel.dist_gpipe_pipeline_async import GpipeAsync
from utils.dist_args_utils import *
from utils.dist_pp_train_utils import *
from comm.init_comm import *


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--mode', type=str, default='1f1b', metavar='S',
                        help='use which mode: gpipe or 1f1b.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    init_communicators(args)

    if get_pipeline_parallel_rank() == 0 or get_pipeline_parallel_rank() == args.world_size-1:
        tokenizer = build_tokenizer(args)
        print("token vocab size:", tokenizer.vocab_size)
        train_data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
        num_classes = 2
        vocab_size = tokenizer.vocab_size
    else:
        train_data_loader = None
        num_classes = 2
        vocab_size = -1

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.mode, " with data parallel.")
    else:
        print("Running ", args.mode, " without data parallel.")

    if args.mode == 'gpipe':
        pipe = GpipeAsync(args, vocab_size, num_classes, device, use_dp)
    elif args.mode == '1f1b':
        pipe = Pipe1F1BAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this mode.")
        assert False

    if args.profiling == 'no-profiling':
        distributed_train_foo_iter(args, pipe, device, train_data_loader)
    else:
        trace_file = './trace_json/gpt3_' + args.mode + get_learning_arguments_str(args) \
                     + get_model_arguments_str(args) + get_dist_arguments_str(args) + '_' + args.profiling + '.json'
        if args.profiling == 'tidy_profiling':
            distributed_train_foo_iter(args, pipe, device, train_data_loader)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                distributed_train_foo_iter(args, pipe, device, train_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False


if __name__ == '__main__':
    main()
