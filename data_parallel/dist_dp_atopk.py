import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import math
from comm.comm_utils import *
from .flatten_utils import flatten_params, flatten_tensors
from compress.fixpoint import *
from compress import flag

import os

if 'SYNC_STEPS' not in os.environ:
    sync_steps = 100
    sync_prob = 1.0 / sync_steps
    
else:
    sync_steps = int(os.environ['SYNC_STEPS'])
    sync_prob = 1.0 / sync_steps
    

@torch.no_grad()
def step_update(self, dp_optimizer=None):
    
    fp16_wrapper = None
    
    if not isinstance(self, torch.optim.AdamW):
        fp16_wrapper, self = self, self.optimizer
        assert isinstance(self, torch.optim.AdamW)
        use_fp16 = (fp16_wrapper is not None)
        assert use_fp16
        
    if fp16_wrapper is not None:
        fp16_wrapper._copy_model_grads_to_optimizer_grads()

        found_inf_flag = fp16_wrapper._unscale_optimizer_grads_and_check_for_nan()
        fp16_wrapper.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            print("!!! Warning: find inf in fp16 optimizer-step() !!!")
            # return False
        
        for params in fp16_wrapper.fp32_from_float16_groups:
            for p in params:
                p.grad = p.grad.nan_to_num()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            
    num_comm = 0

    for group in self.param_groups:
        
        for ii, p in enumerate(group["params"]):
            
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

            state = self.state[p]

            # assert len(state) != 0
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data, dtype=(torch.float16 if use_fp16 else torch.float32))
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["global_p"] = p.data.clone().to(torch.float16 if use_fp16 else torch.float32)
                assert state["global_p"].dtype == torch.float16
                
                grad_buffer, server_error = dp_optimizer._declare_exp_avgs_buffer_by_data(state["global_p"])
                state["extra"] = (grad_buffer, server_error)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
#             if group["correct_bias"]:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
            
            p.data.addcdiv_(exp_avg.to(p.dtype), denom, value=-step_size)

            
            ###
            grad_buffer, server_error = state["extra"]
            
            # p.data /= 2
            # dp_optimizer.dp_comm.all_reduce(p.data)
#             num_comm += p.numel()
#             if dp_optimizer.pp_rank == 1:
#                 print('after allreduce', dp_optimizer.dp_rank, dp_optimizer.pp_rank, 
#                       'step:', state["step"], ii, '/', len(group["params"]), p.numel())
            
#             dp_optimizer.dp_comm.barrier()
                
            # cupy_dp_stream = cupy.cuda.ExternalStream(dp_optimizer.torch_optim_comp_stream.cuda_stream)
            # # dp_optimizer.torch_optim_comp_stream.wait_event(dp_optimizer.backward_ready_event)
            # print('sync_prob', sync_prob)
            # print(p.dtype, p.device, state["global_p"].dtype, state["global_p"].device)
            dp_optimizer.dp_comm.all_reduce_opt_topk_2(
                p.data.view(-1), 
                global_data=state["global_p"].data.view(-1),
                buffer=grad_buffer,
                server_error=server_error,
                # stream=cupy_dp_stream,
                topk_ratio=sync_prob, caller=dp_optimizer
            )
            
            # torch.cuda.synchronize()
            # dp_optimizer.dp_comm.barrier()
            ###
            
            
            if group["weight_decay"] > 0.0:
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                
    # print('COMM:', dp_optimizer.dp_rank, dp_optimizer.pp_rank, 
          # 'step:', state["step"], num_comm)
                
    if fp16_wrapper is not None:
        fp16_wrapper._copy_optimizer_params_to_model_params()
        # Successful update.
        return True
            

class ATopKDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=False):
        assert not flatten
        self.dp_bits = args.dp_bits
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.pp_comm = get_pipeline_parallel_comm()
        self.pp_rank = get_pipeline_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        
        self.mmt_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.mmt_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))

        self.th = torch.zeros(1, dtype=torch.uint8, device=device)
        
        
        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            # assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.gather_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.gather_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.worker_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.worker_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)
            
    def allreduce_parameters(self):
        self._local_parameters_backup = [
            p.data.clone() for p in self.module.parameters()
        ]
        torch.cuda.synchronize()
        self.dp_comm.barrier()
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            # self.dp_comm_stream.wait_event(self.backward_ready_event)
            for name, para in self.module.named_parameters():
                # self.profile_mark_allreduce_start(name)
                para.data /= self.dp_group_size
                self.dp_comm.all_reduce(para.data, stream=cupy_dp_stream)
                # self.profile_mark_allreduce_end(name)
            # self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)
        torch.cuda.synchronize()
        self.dp_comm.barrier()

    def rollback_parameters(self):
        if not hasattr(self, '_local_parameters_backup'):
            return
        
        for p, p_local in zip(self.module.parameters(), self._local_parameters_backup):
            p.data[:] = p_local.data
            
        del self._local_parameters_backup
        
    def _declare_exp_avgs_buffer_by_data(self, data):
        assert data.data.numel() % self.dp_group_size == 0
        chunk_size = data.data.numel() // self.dp_group_size
        topk_ratio = sync_prob
        _data = torch.zeros(chunk_size, device=data.device, dtype=data.dtype)

        _data_compressed = compress_topk(
            _data, int(topk_ratio * _data.numel()))
        grad_buffer = [
            (torch.zeros_like(_data_compressed[0]),
             torch.zeros_like(_data_compressed[1])) for i in range(self.dp_group_size)
        ]
        server_error = torch.zeros_like(_data).half()
        return grad_buffer, server_error
            
    def _sync_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            self.profile_mark_sync_grad_start()
            # step_update_exp_avg(self.optimizer)
            for para in self.module.parameters():
                para.grad /= self.dp_group_size
                self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
            self.profile_mark_allreduce_end()
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
    def optimizer_step(self):
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.sync_gradients_ready_event)
            self.profile_mark_optimizer_step_start()
            step_update(self.optimizer, dp_optimizer=self)
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        # assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(self.sync_gradients_ready_event)*1e+3
        allreduce_log = {"name": "opt_shardedPS_sync", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.sync_gradients_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        
        
        allreduce_slot = self.gather_start_event.elapsed_time(self.gather_end_event)*1e+3
        allreduce_log = {"name": "gather grads", "ph": "X", "pid": self.global_rank, "tid": "9. optimizer-comm",
                         "ts": self.get_ts(self.gather_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.sync_start_event.elapsed_time(self.sync_end_event)*1e+3
        allreduce_log = {"name": "distribute grads", "ph": "X", "pid": self.global_rank, "tid": "10. optimizer-comm",
                         "ts": self.get_ts(self.sync_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.worker_compress_start_event.elapsed_time(self.worker_compress_end_event)*1e+3
        allreduce_log = {"name": "worker compress", "ph": "X", "pid": self.global_rank, "tid": "11. optimizer-comm",
                         "ts": self.get_ts(self.worker_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.server_compress_start_event.elapsed_time(self.server_compress_end_event)*1e+3
        allreduce_log = {"name": "server compress", "ph": "X", "pid": self.global_rank, "tid": "12. optimizer-comm",
                         "ts": self.get_ts(self.server_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        return profiling_log
