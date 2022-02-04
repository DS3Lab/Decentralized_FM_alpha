from comm.init_comm import *


class CentralPS:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = args.enable_tidy_profiling
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.reduce_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.module = module
        if self.dp_rank == 0:
            assert optimizer is not None
            self.optimizer = optimizer
            self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        else:
            self.optimizer = None
            self.optimizer_step_ready_event = None
        num_paras = self._compute_total_para_num()
        print("Total number of parameters: {} of size {} MB".format(num_paras, num_paras*4//1024//1024))

    def _compute_total_para_num(self):
        total_count = 0
        for para in self.module.parameters():
            print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
        return total_count

    def reduce_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            for para in self.module.parameters():
                self.dp_comm.reduce(para.grad,dst=0, stream=cupy_dp_stream)

    def broadcast_parameters(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            if self.dp_rank == 0:
                self.dp_comm_stream.wait_event(self.optimizer_step_ready_event)
            for para in self.module.parameters():
                self.dp_comm.broadcast(para.data, src=0, stream=cupy_dp_stream)

    def optimizer_step(self):
        if self.dp_rank == 0:
            with torch.cuda.stream(self.torch_optim_comp_stream):
                pass
