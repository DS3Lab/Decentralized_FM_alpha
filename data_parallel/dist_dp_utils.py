from .dist_dp_allreduce import AllReduceDP
from .dist_dp_central_ps import CentralPSDP


def get_dp_module(args, device, module, optimizer):
    if args.dp_mode == 'allreduce':
        return AllReduceDP(device, module, optimizer)
    elif args.dp_mode == 'central_ps':
        return CentralPSDP(device, module, optimizer)
    else:
        print("Not recognize this data parallel mode.")
        assert False