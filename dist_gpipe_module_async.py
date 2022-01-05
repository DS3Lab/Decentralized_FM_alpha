import time

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

        self.micro_batch_num = args.batch_size // args.micro_batch_num
        assert (args.batch_size % args.micro_batch_num == 0)
        self.micro_batch_size = args.micro_batch_num
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.forward_comm_ready_events = [torch.cuda.Event(enable_timing=args.timing, blocking=False)
                                          for _ in range(args.micro_batch_num)]
        self.backward_comm_ready_events = [torch.cuda.Event(enable_timing=args.timing, blocking=False)
                                           for _ in range(args.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=args.timing, blocking=False)
                                          for _ in range(args.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=args.timing, blocking=False)
                                           for _ in range(args.micro_batch_num)]
        if args.rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_dim),
                                                    requires_grad=True, device=self.device)
                                        for _ in range(args.micro_batch_num)]
        if args.rank == args.world_size - 1:
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_grad = [torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_dim),
                                                          requires_grad=False, device=self.device)
                                              for _ in range(args.micro_batch_num)]

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

    def forward_stage(self, input_data=None):
        # print("Forward stage start! rank-", self.rank)
        if self.rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        output_micro_batches = []
        for i in range(self.micro_batch_num):
            current_micro_input = self.input_micro_batches[i]
            comm_ready_event = self.forward_comm_ready_events[i]
            comp_ready_event = self.forward_comp_ready_events[i]
            if self.rank == 0:  # Only send output to next node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output = self.model(current_micro_input)
                    self.torch_comp_stream.record_event(comp_ready_event)
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.torch_comm_stream.wait_event(comp_ready_event)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_comm_stream)
            elif self.rank == self.world_size - 1:  # Only receive input from last node, do not send
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.comm.recv(current_micro_input, src=self.pre_node_rank, stream=cupy_comm_stream)
                    self.torch_comm_stream.record_event(comm_ready_event)
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(comm_ready_event)
                    current_micro_output = self.model(current_micro_input)
            else:  # receive, compute, and send
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.comm.recv(current_micro_input, src=self.pre_node_rank, stream=cupy_comm_stream)
                    self.torch_comm_stream.record_event(comm_ready_event)
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(comm_ready_event)
                    current_micro_output = self.model(current_micro_input)
                    self.torch_comp_stream.record_event(comp_ready_event)
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.torch_comm_stream.wait_event(comp_ready_event)
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_comm_stream)
            output_micro_batches.append(current_micro_output)
        return output_micro_batches

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor],
                       loss_func=torch.nn.functional.cross_entropy, target=None):
        # print("Backward stage start! rank-", self.rank)
        if self.rank == self.world_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)

        for i in range(self.micro_batch_num):
            comm_ready_event = self.backward_comm_ready_events[i]
            comp_ready_event = self.backward_comp_ready_events[i]
            if self.rank == self.world_size - 1:  # only send grad back to last node, do not receive
                with torch.cuda.stream(self.torch_comp_stream):
                    loss = loss_func(
                        input=cached_output_micro_batches[i],
                        target=target_as_micro_batches[i].to(self.device))
                    loss.backward()
                    self.torch_comp_stream.record_event(comp_ready_event)
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.torch_comm_stream.wait_event(comp_ready_event)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_comm_stream)
            elif self.rank == 0: # only receive grad from previous node, do not send
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_comm_stream)
                    self.torch_comm_stream.record_event(comm_ready_event)
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(comm_ready_event)
                    cached_output_micro_batches[i].backward(
                        gradient=self.output_micro_batches_grad[i])
            else:  # receive, compute and send
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.comm.recv(self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_comm_stream)
                    self.torch_comm_stream.record_event(comm_ready_event)
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(comm_ready_event)
                    cached_output_micro_batches[i].backward(
                        gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(comp_ready_event)
                with torch.cuda.stream(self.torch_comm_stream):
                    cupy_comm_stream = cupy.cuda.ExternalStream(self.torch_comm_stream.cuda_stream)
                    self.torch_comm_stream.wait_event(comp_ready_event)
                    self.comm.send(self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_comm_stream)

    def sgd_iter(self, input=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        self.optimizer.zero_grad()
        outputs = self.forward_stage(input)
        forward_time = time.time()
        print("Rank {} node forward pass takes {:3.2f}s".format(self.rank,  forward_time-start_time))
        self.backward_stage(outputs, target)
        backward_time = time.time()
        print("Rank {} node backward pass takes {:3.2f}s".format(self.rank,  backward_time-forward_time))
        self.optimizer.step()
        end_time = time.time()
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.rank, end_time - start_time))