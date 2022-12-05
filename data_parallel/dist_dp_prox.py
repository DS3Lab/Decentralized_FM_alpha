import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import math
from comm.comm_utils import *
from .flatten_utils import flatten_params, flatten_tensors
from compress.fixpoint import *
from compress import flag


sync_prob = 0.02


class ProxDP:
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
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.allreduce_grad_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

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
            
    def pre_optimization(self):
        if self.global_rank == 0:
            self.th.data[0] = (torch.rand(1) > 1 - sync_prob)
        if self.dp_rank == 0:
            # sync to all nodes of dp_rank0 from global_rank0
            self.pp_comm.broadcast(self.th, 0)
        # sync to all nodes from dp_rank0
        self.dp_comm.broadcast(self.th, 0)
        
        if self.th.item() == 1:
            print('do sync !!!!!!!!')
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
            
            if hasattr(self.optimizer, 'reload_model_params'):
                print('reloading model params')
                self.optimizer.reload_model_params()
            
    def optimizer_step(self):
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.allreduce_grad_ready_event)
            self.profile_mark_optimizer_step_start()
            self.pre_optimization()
            self.optimizer.step()
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
