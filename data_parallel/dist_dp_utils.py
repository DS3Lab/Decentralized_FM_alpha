from .dist_dp_allreduce import AllReduceDP
from .dist_dp_central_ps import CentralPSDP
from .dist_dp_sharded_ps import ShardedPSDP
from .dist_dp_sharded_ps_topk import ShardedPSDPTopK
from .dist_dp_sharded_ps_compressed import ShardedPSDPCompressed
from .dist_dp_local import LocalDP
from .dist_dp_prox import ProxDP
from .dist_dp_aproxadam import AProxAdamDP
from .dist_dp_afreeze import AFreezeDP
from .dist_dp_atopk import ATopKDP
from .dist_dp_proxskip import ProxSkipDP
from .dist_dp_proxskip_adam import ProxSkipAdamDP


def get_dp_module(args, device, module, optimizer):
    print("Data parallel implementation: ", args.dp_mode)
    if args.dp_mode == 'allreduce':
        return AllReduceDP(args, device, module, optimizer, flatten=False) # flatten seems to be not compatible with fp16
    elif args.dp_mode == 'central_ps':
        return CentralPSDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'sharded_ps':
        return ShardedPSDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'sharded_ps_topk':
        return ShardedPSDPTopK(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'sharded_ps_compressed':
        return ShardedPSDPCompressed(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'local':
        return LocalDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'prox':
        return ProxDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'proxskip':
        return ProxSkipDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'proxadam':
        return ProxSkipAdamDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'aproxadam':
        return AProxAdamDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'afreeze':
        return AFreezeDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'atopk':
        return ATopKDP(args, device, module, optimizer, flatten=False)
    else:
        print("Not recognize this data parallel mode.")
        assert False
