import argparse
import time
import random
import numpy as np
import torch
import torch.autograd.profiler as profiler
from task_datasets.wikitext import get_wikitext_train_data_loader, get_wikitext_test_data_loader
from task_datasets.wiki103 import get_wiki103_train_data_loader, get_wiki103_test_data_loader
from task_datasets.arxiv21 import get_arxiv21_train_data_loader, get_arxiv21_test_data_loader
from task_datasets.openwebtext import get_openwebtext_train_data_loader
from modules.hf_gpt2_train_module import GPTConfig
from pipeline_parallel.dist_pp_utils import get_pp_finetune_module as get_pp_module

from transformers import AutoTokenizer

try:
    import wandb
except Exception as e:
    wandb = None
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from utils.dist_test_utils import *
from comm.comm_utils import *
from coordinator.coordinate_client import *

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
    for e in range(args.n_epochs):
        
        distributed_train_lm_iter(args, pipe, device, train_data_loader)
        
        if test_data_loader is not None and args.do_evaluation:
            distributed_test_lm_iter(args, pipe, device, test_data_loader)
            
        if get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
            if wandb is not None:
                wandb.log({'epoch': e}, step=pipe.global_step)

def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_torch_distributed_w_coordinator_arguments(parser)
    add_device_arguments(parser)
    # add_torch_distributed_arguments(parser)
    add_training_model_arguments(parser)
    # add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    parser.add_argument('--model-name', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--task-name', type=str, default='wikitext', metavar='S',
                        help='task name')
    parser.add_argument('--task-type', type=str, default='language_model', metavar='S',
                        help='task typw')
    parser.add_argument('--n-epochs', type=int, default=10, help='-')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='-')
    parser.add_argument('--warmup-steps', type=int, default=None, help='-')
    parser.add_argument('--total-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--do-evaluation', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='do evaluation or not.')
    args = parser.parse_args()
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    coord_client = CoordinatorTrainClient(args)
    prime_ip, rank = coord_client.notify_train_join()
    print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")

    args.rank = rank
    init_communicators_with_coordinator(args, prime_ip, rank)
    
    config = GPTConfig.from_pretrained(args.model_name)
#     config.attn_pdrop = 0.0
#     config.embd_pdrop = 0.0
#     config.resid_pdrop = 0.0
#     config.summary_first_dropout = 0.0
    
    config.n_layer = args.num_layers
#     if get_pipeline_parallel_rank() == args.pipeline_group_size-1:
#         args.num_layers -= 3
#         config.n_layer = args.num_layers  # num_layers per node
#     elif get_pipeline_parallel_rank() == args.pipeline_group_size-4:
#         args.num_layers += 1
#         config.n_layer = args.num_layers  # num_layers per node
#     elif get_pipeline_parallel_rank() == args.pipeline_group_size-3:
#         args.num_layers += 1
#         config.n_layer = args.num_layers  # num_layers per node
#     elif get_pipeline_parallel_rank() == args.pipeline_group_size-2:
#         args.num_layers += 1
#         config.n_layer = args.num_layers  # num_layers per node
#     else:
#         config.n_layer = args.num_layers  # num_layers per node
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.model_max_length = args.seq_length
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", tokenizer.vocab_size)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.task_name == 'wikitext':
        train_data_loader = get_wikitext_train_data_loader(args, tokenizer)
        test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
    elif args.task_name == 'wiki103':
        train_data_loader = get_wiki103_train_data_loader(args, tokenizer)
        test_data_loader = get_wiki103_test_data_loader(args, tokenizer)
    elif args.task_name == 'arxiv21':
        train_data_loader = get_arxiv21_train_data_loader(args, tokenizer)
        test_data_loader = get_arxiv21_test_data_loader(args, tokenizer)
    elif args.task_name == 'openwebtext':
        train_data_loader = get_openwebtext_train_data_loader(args, tokenizer)
        test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
    else:
        raise Exception('unknown task.')
        
    if args.warmup_steps is None:
        args.warmup_steps = len(train_data_loader)
    if args.total_steps is None:
        args.total_steps = len(train_data_loader) * args.n_epochs
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
    
    print('initializing pipeline')
    pipe = get_pp_module(args, config, device, use_dp)
    
    if args.load_pretrained_model:
        print('loading model')
        if get_pipeline_parallel_rank() == 0:
            pipe.model.model[0].load_state_dict(
                torch.load(f'{args.model_name}/pytorch_embs.pt')
            )
            for i in range(len(pipe.model.model)-1):
                print(i)
                pipe.model.model[i+1].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{i}.pt')
                )
        elif get_pipeline_parallel_rank() == args.pipeline_group_size-1:
            _i = get_pipeline_parallel_rank() * args.num_layers
            for i in range(len(pipe.model.model)-1):
                print(_i + i)
                pipe.model.model[i].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
                )
            pipe.model.model[-1].load_state_dict(
                torch.load(f'{args.model_name}/pytorch_lm_head.pt')
            )
        else:
            _i = get_pipeline_parallel_rank() * args.num_layers
            for i in range(len(pipe.model.model)):
                print(_i + i)
                pipe.model.model[i].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
                )      

    print('starting train loop....')
    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            except Exception as e:
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    print(get_pipeline_parallel_rank(), 'finished.')
    train_finish_msg = str(rank) + '#' + str(round(0.0, 3))
    coord_client.notify_train_finish(message=train_finish_msg)

if __name__ == '__main__':
    main()
