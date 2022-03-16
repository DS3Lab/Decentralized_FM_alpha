import torch.cuda
from comm.init_comm import *


class AllReduceDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.allreduce_grad_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.module = module
        if self.dp_rank == 0:
            assert optimizer is not None
            self.optimizer = optimizer
            self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        else:
            self.optimizer = None
            self.optimizer_step_ready_event = None
        num_paras = self._compute_total_para_num()
        print("Total number of parameters: {} of size {} MB".format(num_paras, num_paras * 4 // 1024 // 1024))
        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None
            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling,
                                                               blocking=False)
            self.allreduce_grad_start_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling,
                                                               blocking=False)

    def _compute_total_para_num(self):
        total_count = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
        return total_count

    def profile_mark_allreduce_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.allreduce_grad_start_event)

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling and self.dp_rank == 0:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)

    def profile_mark_allreduce_end(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)

    def _allreduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            self.profile_mark_allreduce_start()
            for para in self.module.parameters():
                self.dp_comm.all_reduce(para.grad, stream=cupy_dp_stream)
            self.dp_comm_stream.record_event(self.allreduce_grad_ready_event)

    def optimizer_step(self):
        self._allreduce_gradients()
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.allreduce_grad_ready_event)
            self.profile_mark_optimizer_step_start()
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
        allreduce_slot = self.allreduce_grad_start_event.elapsed_time(self.allreduce_grad_ready_event) * 1e+3
        allreduce_log = {"name": "opt_reduce", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.allreduce_grad_start_event), "dur": allreduce_slot,
                         "cname": "cq_build_passed"}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        return profiling_log
