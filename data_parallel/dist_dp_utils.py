from .dist_dp_allreduce import AllReduceDP
from .dist_dp_central_ps import CentralPSDP
from .dist_dp_sharded_ps import ShardedPSDP
from .dist_dp_sharded_ps_topk import ShardedPSDPTopK
from .dist_dp_sharded_ps_compressed import ShardedPSDPCompressed
from .dist_dp_local import LocalDP
from .dist_dp_prox import ProxDP
from .dist_dp_afreeze import AFreezeDP
from .dist_dp_atopk import ATopKDP
from .dist_dp_proxskip import ProxSkipDP
from .dist_dp_proxskip_adam import ProxSkipAdamDP
from .dist_dp_afreeze_compressed import AFreezeCompressDP
from .dist_dp_afreeze_compressed_2 import AFreezeCompress2DP
from .dist_dp_slot_sgd import SlotSGDDP
from .dist_dp_slot_sgd_gloo import SlotSGDGlooDP
from .dist_dp_slot_sgd_gloo_replacement import SlotSGDGlooRDP
from .dist_dp_fake_slot_sgd_gloo import FakeSlotSGDGlooDP
from .dist_dp_slot_sgd_benchmark import SlotSGDBenchDP
from .dist_dp_slot_sgd_gloo_benchmark import SlotSGDGlooBenchDP
from .dist_dp_sharded_ps_quant import ShardedPSDPQuant
from .dist_dp_powersgd import PowerSGDDP
from .dist_dp_qsl import QSLDP


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
    elif args.dp_mode == 'sharded_ps_quant':
        return ShardedPSDPQuant(args, device, module, optimizer, flatten=False)
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
    elif args.dp_mode == 'afreeze':
        return AFreezeDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'afreeze_compressed':
        return AFreezeCompressDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'afreeze_compressed_2':
        return AFreezeCompress2DP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'atopk':
        return ATopKDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'slot_sgd':
        return SlotSGDDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'slot_sgd_gloo':
        return SlotSGDGlooDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'slot_sgd_bench':
        return SlotSGDBenchDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'slot_sgd_gloo_bench':
        return SlotSGDGlooBenchDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'slot_sgd_gloo_replacement':
        return SlotSGDGlooRDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'fake_slot_sgd_gloo':
        return FakeSlotSGDGlooDP(args, device, module, optimizer, flatten=True)
    elif args.dp_mode == 'powersgd':
        return PowerSGDDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'qsl':
        return QSLDP(args, device, module, optimizer, flatten=True)
    else:
        print("Not recognize this data parallel mode.")
        assert False
