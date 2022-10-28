import argparse
import time
import random
import numpy as np
import torch
import torch.autograd.profiler as profiler
# from tasks.data_loaders.openwebtext_prefix import get_openwebtext_train_data_loader as get_openwebtext_prefix_train_data_loader
from tasks.data_loaders.pile import get_pile_train_data_loader
from tasks.data_loaders.c4 import get_c4_train_data_loader
from modules.utils import gpt_loss_func
from modules.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module

from transformers import AutoConfig

from coordinator.http_coordinate_client import get_coordinator_client, init_coordinator_client

import wandb
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from utils.dist_test_utils import *
from utils.dist_checkpoint_utils import *
from comm.comm_utils import *
import compress.flag

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
#     for e in range(args.n_epochs):
#         if e < args.warmup_epochs:
#             compress.flag.FLAG_DISABLE_COMPRESSION = True
#         else:
#             compress.flag.FLAG_DISABLE_COMPRESSION = False
            
#         distributed_train_lm_iter(args, pipe, device, train_data_loader)
        
#         if test_data_loader is not None and args.do_evaluation:
#             distributed_test_lm_iter(args, pipe, device, test_data_loader)

    pipe.model.train() # Flag .training to True to enable Dropout
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    pp_comm = get_pipeline_parallel_comm()
    
    stop_flag = torch.zeros(1, dtype=torch.int64).to(device)
    
    input_ids = torch.zeros(
        [args.batch_size, args.seq_length], 
        dtype=torch.int64
    ).to(device)
    
    masks = torch.zeros(
        [args.batch_size, args.seq_length], 
        dtype=torch.uint8
    ).to(device)
    
    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:
        
        for data in train_data_loader:
            # if i < pipe.global_step:
            #     continue
                
            if use_dp:
                dp_comm.broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
            
            input_ids_global = data['input_ids'].to(torch.int64).to(device)
            masks_global = torch.ones_like(input_ids_global).to(torch.uint8).to(device)
            
            # FLM said dynamic sampling random ratio is better than static
            th = (torch.rand(1) * 0.15).item()
            masks_global[torch.rand(masks_global.shape) < th] = 0
            
            input_ids_list = input_ids_global.chunk(dp_size)
            masks_list = masks_global.chunk(dp_size)
            
            if use_dp:
                for j in range(1, dp_size):
                    dp_comm.send(
                        input_ids_list[j], j,
                    )
                    dp_comm.send(
                        masks_list[j], j,
                    )
                
            input_ids = input_ids_list[0]
            masks = masks_list[0]
            
            pp_comm.broadcast(input_ids, 0)
            pp_comm.broadcast(masks, 0)
            
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(input_ids, None, aux_input_data={'mask': masks})
            
            if pipe.global_step % args.checkpoint_steps == 0 and dp_rank == 0:
                save_checkpoint(pipe, args)
            
            if pipe.global_step >= args.total_steps:
                stop_flag.data[:] = 1
            
    elif get_pipeline_parallel_rank() == 0:
        
        while True:
            
            dp_comm.broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            dp_comm.recv(
                input_ids, 0,
            )
            dp_comm.recv(
                masks, 0,
            )
            pp_comm.broadcast(input_ids, 0)
            pp_comm.broadcast(masks, 0)
            
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(input_ids, None, aux_input_data={'mask': masks})
            
            if pipe.global_step % args.checkpoint_steps == 0 and dp_rank == 0:
                save_checkpoint(pipe, args)
            
            
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        
        while True:
            
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            pp_comm.broadcast(input_ids, 0)
            pp_comm.broadcast(masks, 0)
            labels = input_ids.clone()
            labels[:, 1:][(1-masks).bool()[:, :-1]] = -100 # mask the next token of masked tokens
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func, aux_input_data={'mask': masks}) # lm loss func
            
            if pipe.global_step % args.checkpoint_steps == 0 and dp_rank == 0:
                save_checkpoint(pipe, args)
        
    else:
        
        while True:
            
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            pp_comm.broadcast(input_ids, 0)
            pp_comm.broadcast(masks, 0)
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(None, None, aux_input_data={'mask': masks})
            
            if pipe.global_step % args.checkpoint_steps == 0 and dp_rank == 0:
                save_checkpoint(pipe, args)
        

def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_acitvation_compression_arguments(parser)
    parser.add_argument('--model-name', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--model-type', type=str, default='gptj', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--checkpoint-path', type=str, default='model_checkpoints/gpt2')
    parser.add_argument('--task-name', type=str, default='wikitext', metavar='S',
                        help='task name')
    parser.add_argument('--warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--train-warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--total-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--load-checkpoint', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='no-profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--evaluation-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, do evaluation. (0 means do not do evaluation)')
    parser.add_argument('--checkpoint-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, save checkpoint. (0 means do not save checkpoint)')
    parser.add_argument('--net-interface', 
                        type=str, default='lo', metavar='S',
                        help='net_interface')
    parser.add_argument('--job-id', 
                        type=str, default="0", metavar='S',
                        help='an uuid')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
        
    if args.job_id != "0":
        init_coordinator_client(args, None)
        coord_client = get_coordinator_client()
        res = coord_client.notify_inference_join(args.net_interface)
        prime_ip = res['prime_ip']
        rank = res['rank']
        port = res['nccl_port']

        print(f"job id: {args.job_id}")
        print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")

        args.dist_url = f"tcp://{prime_ip}:{port}"
        args.rank = rank

    init_communicators(args)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    
    config = AutoConfig.from_pretrained(args.model_name)
    
    # num layer globally
    args.max_layers = config.num_layers if hasattr(config, 'num_layers') else config.n_layer
    
    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    # config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", config.vocab_size)
    
    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:
        if args.task_name == 'pile':
            train_data_loader = get_pile_train_data_loader(args, tokenizer)
            test_data_loader = None #get_wikitext_test_data_loader(args, tokenizer)
        elif args.task_name == 'c4':
            train_data_loader = get_c4_train_data_loader(args, tokenizer)
            test_data_loader = None #get_wikitext_test_data_loader(args, tokenizer)  
        else:
            raise Exception('unknown task.')
    else:
        train_data_loader = None
        test_data_loader = None
        
    if args.total_steps is None:
        args.total_steps = len(train_data_loader)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")
    
    pipe = get_pp_module(args, config, device, use_dp)
    
    if args.load_checkpoint:
        load_checkpoint(pipe, args)

    if args.fp16:
        pipe.optimizer.reload_model_params()

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

if __name__ == '__main__':
    main()
