import torch

from .powersgd import Aggregator, AllReduce, Config, PowerSGD
from .utils import params_in_optimizer
from comm.comm_utils import *


def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    # params = params_in_optimizer(optimizer)
#     grads = [p.grad.data for p in params]  # type: ignore
#     avg_grads = aggregator.aggregate(grads)  # subtracts the approximation from grads

#     # Temporarily set parameter's gradients to the aggregated values
#     for (p, g) in zip(params, avg_grads):
#         p.grad = g

    # Run an optimizer step
    # success = optimizer.step()
    
    comm = get_data_parallel_comm()
    
    optimizer._copy_model_grads_to_optimizer_grads()

    found_inf_flag = optimizer._unscale_optimizer_grads_and_check_for_nan()
    t_found_inf_flag = torch.tensor(found_inf_flag).long().cuda()
    comm.all_reduce(t_found_inf_flag)
    found_inf_flag = (t_found_inf_flag.item() > 0)
    
    optimizer.grad_scaler.update(found_inf_flag)

    # If we found inf/nan, skip the update.
    if found_inf_flag:
        print("!!! Warning: find inf in fp16 optimizer-step() !!!")
        optimizer.zero_grad(set_to_none=False)
        
    else:
        
        params = []
        for optimizer_group in optimizer.fp32_from_float16_groups:
            for optimizer_param in optimizer_group:
                params.append(optimizer_param)
                
        params_fp16 = []
        for optimizer_group in optimizer.float16_groups:
            for optimizer_param in optimizer_group:
                params_fp16.append(optimizer_param)
                
        grads = [p.grad.data for p in params]
        
        #for (p, p16, g) in zip(params, params_fp16, grads):
        #    assert p.shape == g.shape
        
        avg_grads = aggregator.aggregate(grads)
        
        #for (p, p16, g) in zip(params, params_fp16, grads):
        #    assert p.shape == g.shape
        
        for (p, g) in zip(params, avg_grads):
            p.grad.data[:] = g
    
        for _params in optimizer.fp32_from_float16_groups:
            torch.nn.utils.clip_grad_norm_(_params, 1.0)

        # Step the optimizer.
        optimizer.optimizer.step()

        for (p, p16, g) in zip(params, params_fp16, grads):
            assert p.shape == g.shape
            p.grad.data[:] = g
            p16.grad.data[:] = g.half()
            
        optimizer._copy_optimizer_params_to_model_params()