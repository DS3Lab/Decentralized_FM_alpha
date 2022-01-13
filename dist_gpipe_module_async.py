import time
import json
import torch.nn.functional
from torch import optim
from dist_gpt_utils import *
from dist_gpt_module import *


class GpipeAsync:
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
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

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
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_grad = [torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                                          requires_grad=False, device=self.device)
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

    def get_forward_ts(self, event):
        return self.forward_init_time_stamp + self.forward_init_event.elapsed_time(event) * 1e+3

    def get_backward_ts(self, event):
        return self.backward_init_time_stamp + self.backward_init_event.elapsed_time(event) * 1e+3

    def forward_stage(self, input_data=None):
        # print("Forward stage start! rank-", self.rank)
        if self.rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        output_micro_batches = []
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.forward_init_time_stamp = time.time() * 1e+6
            self.forward_init_event.record()
        for i in range(self.micro_batch_num):
            current_micro_input = self.input_micro_batches[i]
            if self.rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(current_micro_input)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_comm_stream)
                    self.profile_mark_forward_send_end(i)
            elif self.rank == self.world_size - 1:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(current_micro_input, src=self.pre_node_rank, stream=cupy_comm_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(current_micro_input)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_forward_recv_start(i)
                    self.comm.recv(current_micro_input, src=self.pre_node_rank, stream=cupy_comm_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(current_micro_input)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.profile_mark_forward_send_start(i)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_comm_stream)
                    self.profile_mark_forward_send_end(i)
            output_micro_batches.append(current_micro_output)
        if self.enable_tidy_profiling:
            self.profiling_forward_stage()
        return output_micro_batches

    def profiling_forward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.rank != 0:
                recv_slot = self.forward_recv_start_events[i].elapsed_time(self.forward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.rank, "tid": "1. forward-recv",
                            "ts": self.get_forward_ts(self.forward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_comp_start_events[i].elapsed_time(self.forward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.rank, "tid": "2. forward-compute",
                        "ts": self.get_forward_ts(self.forward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            print(comp_log)
            self.profiling_log.append(comp_log)

            if self.rank != self.world_size - 1:
                send_slot = self.forward_send_start_events[i].elapsed_time(self.forward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.rank, "tid": "3. forward-send",
                            "ts": self.get_forward_ts(self.forward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                print(send_log)
                self.profiling_log.append(send_log)

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None,
                       loss_func=torch.nn.functional.cross_entropy):
        # print("Backward stage start! rank-", self.rank)
        if self.rank == self.world_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.backward_init_time_stamp = time.time() * 1e+6
            self.backward_init_event.record()
        for i in range(self.micro_batch_num):
            if self.rank == self.world_size - 1:  # only send grad back to last node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_backward_comp_start(i)
                    loss = loss_func(input=cached_output_micro_batches[i], target=target_as_micro_batches[i])
                    loss.backward()
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_comm_stream)
                    self.profile_mark_backward_send_end(i)
            elif self.rank == 0:  # only receive grad from previous node, do not send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_comm_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:  # receive, compute and send
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.profile_mark_backward_recv_start(i)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_comm_stream)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    self.profile_mark_backward_comp_start(i)
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.profile_mark_backward_send_start(i)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_comm_stream)
                    self.profile_mark_backward_send_end(i)
        if self.enable_tidy_profiling:
            self.profiling_backward_stage()

    def profiling_backward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.rank != self.world_size - 1:
                recv_slot = self.backward_recv_start_events[i].elapsed_time(self.backward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.rank, "tid": "4. backward-recv",
                            "ts": self.get_backward_ts(self.backward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}
                print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.backward_comp_start_events[i].elapsed_time(self.backward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.rank, "tid": "5. backward-compute",
                        "ts": self.get_backward_ts(self.backward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            print(comp_log)
            self.profiling_log.append(comp_log)
            if self.rank != 0:
                send_slot = self.backward_send_start_events[i].elapsed_time(self.backward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.rank, "tid": "6. backward-send",
                            "ts": self.get_backward_ts(self.backward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                print(send_log)
                self.profiling_log.append(send_log)

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
        optimizer_log = {"name": "opt", "ph": "X", "pid": self.rank, "tid": "7. optimizer step",
                         "ts": self.optimizer_start_time_stamp, "dur": optimizer_slot, "cname": "memory_dump"}
        print(optimizer_log)
        self.profiling_log.append(optimizer_log)

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def sgd_iter(self, input_=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        self.optimizer.zero_grad()
        outputs = self.forward_stage(input_)
        forward_time = time.time()
        print("Rank {} node forward pass takes {:3.2f}s".format(self.rank,  forward_time-start_time))
        self.comm.barrier()  # This is an educated guess that such barrier would make it fair TC (probably required)
        self.backward_stage(outputs, target)
        backward_time = time.time()
        print("Rank {} node backward pass takes {:3.2f}s".format(self.rank,  backward_time-forward_time))
        self.optimizer_step()
        torch.cuda.synchronize()
        end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.rank, iter_time))
        return iter_time
