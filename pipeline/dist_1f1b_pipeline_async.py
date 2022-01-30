import time
import json
import torch.nn.functional
from torch import optim
from comm.nccl_backend import *
from modules.dist_gpt_pp_module import *


class Pipe1F1BAsync:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, vocab_size, num_classes, device):
        self.world_size = args.world_size
        self.rank = args.rank
        self.pre_node_rank = args.rank - 1
        self.post_node_rank = args.rank + 1 if args.rank != args.world_size - 1 else -1
        self.comm = init_comm(args)

        assert (args.batch_size % args.micro_batch_size == 0)
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_forward_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_backward_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_forward_send_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_backward_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]

        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                            for _ in range(self.micro_batch_num)]

            self.backward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                             for _ in range(self.micro_batch_num)]
            self.forward_init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.forward_init_time_stamp = None
            self.backward_init_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.backward_init_time_stamp = None
            self.optimizer_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_start_time_stamp = None

        if args.rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                    requires_grad=True, device=self.device)
                                        for _ in range(self.micro_batch_num)]
        if args.rank == args.world_size - 1:
            self.output_micro_batches = None
        else:
            self.output_micro_batches = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                     requires_grad=True, device=self.device)
                                         for _ in range(self.micro_batch_num)]

        if self.rank == 0:
            self.model = GPTShardFirst(args, vocab_size, num_classes, device)
        elif self.rank == self.world_size - 1:
            self.model = GPTShardLast(args, vocab_size, num_classes, device)
        else:
            self.model = GPTShardMiddle(args, vocab_size, num_classes, device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

    def zero_input_grad(self):
        if self.input_micro_batches:
            for input_micro_batch in self.input_micro_batches:
                if input_micro_batch.grad is not None:
                    input_micro_batch.grad.zero_()

    def profile_mark_forward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.forward_comp_start_events[i])

    def profile_mark_forward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.forward_recv_start_events[i])

    def profile_mark_forward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_start_events[i])

    def profile_mark_forward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.forward_send_end_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(self.backward_comp_start_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_start_events[i])

    def profile_mark_backward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(self.backward_send_end_events[i])

    def forward_backward_stages(self, input_data=None, target=None, loss_func=torch.nn.functional.cross_entropy):
        # TODO this loading part should be updated later
        if self.rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        elif self.rank == self.world_size - 1:
            assert (target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert (input_data is None and target is None)

        # start phase to fill the pipeline
        for i in range(self.world_size - 1 - self.rank):
            if self.rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    self.output_micro_batches[i] = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(self.output_micro_batches[i].data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)
            elif self.rank != self.world_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    self.output_micro_batches[i] = self.model(self.input_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(self.output_micro_batches[i].data, dst=self.post_node_rank, stream=cupy_send_stream)
                    self.profile_mark_forward_send_end(i)




    def optimizer_step(self):
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.optimizer_start_time_stamp = time.time() * 1e+6
        with torch.cuda.stream(self.torch_comp_stream):
            if self.enable_tidy_profiling:
                self.optimizer_start_event.record()
            self.optimizer.step()
            if self.enable_tidy_profiling:
                self.optimizer_end_event.record()
        if self.enable_tidy_profiling:
            self.profiling_optimizer_step()

    def profiling_optimizer_step(self):
        torch.cuda.synchronize()
        optimizer_slot = self.optimizer_start_event.elapsed_time(self.optimizer_end_event) * 1e+3
        optimizer_log = {"name": "opt", "ph": "X", "pid": self.rank, "tid": "7. optimizer-step",
                         "ts": self.optimizer_start_time_stamp, "dur": optimizer_slot, "cname": "bad"}
        print(optimizer_log)
        self.profiling_log.append(optimizer_log)

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)