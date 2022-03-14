import torch


def print_cuda_memory(args, info: str, device=None):
    if args.debug_mem:
        if device is None:
            device = torch.device('cuda', args.cuda_id)
        print("<{}>: current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            info, torch.cuda.memory_allocated(device)/1048576, torch.cuda.max_memory_allocated(device)/1048576))

