import argparse
import time
import torch
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel
# from torch.distributed.fsdp import FullyShardedDataParallel
from fairscale.nn.checkpoint import checkpoint_wrapper
from glue_dataset.qqp import get_glue_qqp_train_data_loader
from glue_dataset.tokenizer import build_tokenizer
from utils.dist_args_utils import *
from utils.dist_debug_utils import *
from modules.gpt_modules import GPTGlueModel, GPTGlueFSDPModel, get_position_id


def main():
    parser = argparse.ArgumentParser(description='Fairscale-ZeRO_S3-GPT3')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser.add_argument('--fsdp-degree', type=str, default="recursive", metavar='S',
                        help='how to use FSDP (default: recursive)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    print("batch size: ", args.batch_size)
    train_dataloader = get_glue_qqp_train_data_loader(args, tokenizer)
    vocab_size = tokenizer.vocab_size
    num_classes = 2

    # disable my own checkpoint
    if args.fsdp_degree == 'simple':
        model = GPTGlueModel(args, vocab_size, num_classes, use_checkpoint=False).to(device)
    elif args.fsdp_degree == 'recursive':
        model = GPTGlueFSDPModel(args, vocab_size, num_classes, use_checkpoint=False).to(device)
    else:
        print("Illegal FSDP degree!")
        assert False
    model = checkpoint_wrapper(model, offload_to_cpu=True)
    torch.cuda.set_device(args.cuda_id)
    dist_model = FullyShardedDataParallel(model, move_params_to_cpu=False, mixed_precision=False)
    print_cuda_memory(args, "Declared FSDP model-"+args.fsdp_degree, device)
    # dist_model = checkpoint_wrapper(dist_model, offload_to_cpu=True)
    optimizer = torch.optim.SGD(dist_model.parameters(), lr=args.lr)
    print_cuda_memory(args, "Declared optimizer for FSDP model", device)
    dist_model.train()

    total_time = 0
    for i, data in enumerate(train_dataloader):
        start_time = time.time()
        input_ids = data['text'].to(device)
        position_ids = get_position_id(args.seq_length, args.batch_size, device)
        labels = data['label'].to(device)
        dist_model.zero_grad()

        output = dist_model(input_ids, position_ids)
        loss = torch.nn.functional.cross_entropy(output, labels)
        forward_time = time.time()
        print("Forward pass takes {:3.2f}s, loss: ".format(forward_time - start_time), loss.item())
        print_cuda_memory(args, "FSDP forward iter is done", device)
        loss.backward()
        backward_time = time.time()
        print("Backward pass takes {:3.2f}s".format(backward_time - forward_time))
        print_cuda_memory(args, "FSDP backward iter is done", device)
        optimizer.step()
        end_time = time.time()
        iter_time = end_time - start_time
        print("Whole iteration takes {:3.2f}s".format(iter_time))
        print_cuda_memory(args, "FSDP optimizer step is done")
        # print(data)
        if i != 0:
            total_time += iter_time
        if i >= args.num_iters - 1:
            break
    averaged_time = total_time / (args.num_iters - 1)
    print("Finished running ", args.num_iters, " iters, averaged run (exclude the first iter) time:", averaged_time)


if __name__ == '__main__':
    main()
