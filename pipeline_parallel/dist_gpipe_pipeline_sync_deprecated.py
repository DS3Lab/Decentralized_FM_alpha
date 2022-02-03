import time
from torch import optim
from utils.dist_args_utils import *
from modules.gpt_modules import *


class GpipeSync:
    def __init__(self, args, ntokens, device):
        self.world_size = args.pipeline_group_size
        self.rank = args.rank
        self.pre_node_rank = args.rank - 1
        self.post_node_rank = args.rank + 1 if args.rank != args.pipeline_group_size - 1 else -1
        self.comm = init_comm(args)

        self.micro_batch_num = args.micro_batch_num
        assert (args.batch_size % args.micro_batch_num == 0)
        self.micro_batch_size = args.batch_size // args.micro_batch_num
        self.seq_length = args.seq_length
        self.embedding_size = args.embedding_dim
        self.ntokens = ntokens

        self.device = device
        if self.rank == 0:
            self.model = GPTShardFirst(args, ntokens, device)
        elif self.rank == self.world_size - 1:
            self.model = GPTShardLast(args, ntokens, device)
        else:
            self.model = GPTShardMiddle(args, ntokens, device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

    def forward_stage(self, input_data=None):
        # print("Forward stage start! rank-", self.rank)
        if self.rank == 0:
            assert(input_data is not None)
            input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        else:
            assert(input_data is None)
            input_micro_batches = []
            for _ in range(self.micro_batch_num):
                if type(self.comm) is NCCLCommunicator:
                    input_micro_batch = torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_size),
                                                    requires_grad=True, device=self.device)
                else:
                    input_micro_batch = torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_size),
                                                    requires_grad=True, device='cpu')
                input_micro_batches.append(input_micro_batch)
        output_micro_batches = []
        for current_micro_input in input_micro_batches:
            if self.pre_node_rank != -1:  # <=> rank != 0
                self.comm.recv(current_micro_input, src=self.pre_node_rank)
                # print("Rank ", self.rank, " recv in forward is doen")
            current_micro_output = self.model(current_micro_input)
            if self.post_node_rank != -1:
                self.comm.send(current_micro_output.data, dst=self.post_node_rank)
            output_micro_batches.append(current_micro_output)
        return input_micro_batches, output_micro_batches

    def backward_stage(self, cached_input_micro_batches: List[torch.Tensor],
                       cached_output_micro_batches: List[torch.Tensor], target=None):
        # print("Backward stage start! rank-", self.rank)
        if self.rank == self.world_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)
        # TODO:
        #  In the figure of GPipe paper and the implementation of TeraPipe, the backward pass is executed in
        #  the reversed order in terms of the micro_batch, I guess this is related to activation recompute,
        #  the last micro-batch does not need to be recomputed in this case.
        #  The current implementation might be modified later.
        for micro_batch_index in range(self.micro_batch_num):
            cached_input_micro_batches[micro_batch_index].to(self.device)
            if self.post_node_rank != -1:  # <=> rank != world_size - 1
                if type(self.comm) is NCCLCommunicator:
                    output_micro_batch_grad = \
                        torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_size), device=self.device)
                else:
                    output_micro_batch_grad = torch.zeros((self.seq_length, self.micro_batch_size, self.embedding_size))
                self.comm.recv(output_micro_batch_grad, src=self.post_node_rank)
                cached_output_micro_batches[micro_batch_index].to(self.device).backward(
                    gradient=output_micro_batch_grad.to(self.device))
            else:
                loss = torch.nn.functional.cross_entropy(
                    input=cached_output_micro_batches[micro_batch_index].to(self.device),
                    target=target_as_micro_batches[micro_batch_index].to(self.device))
                loss.backward()
            if self.pre_node_rank != -1:
                if type(self.comm) is NCCLCommunicator:
                    self.comm.send(cached_input_micro_batches[micro_batch_index].grad, dst=self.pre_node_rank)
                else:
                    self.comm.send(cached_input_micro_batches[micro_batch_index].cpu().grad, dst=self.pre_node_rank)

    def sgd_iter(self, input=None, target=None):
        self.comm.barrier()
        start_time = time.time()
        self.optimizer.zero_grad()
        inputs, outputs = self.forward_stage(input)
        forward_time = time.time()
        print("Rank {} node forward pass takes {:3.2f}s".format(self.rank,  forward_time-start_time))
        self.backward_stage(inputs, outputs, target)
        backward_time = time.time()
        print("Rank {} node backward pass takes {:3.2f}s".format(self.rank,  backward_time-forward_time))
        self.optimizer.step()
        end_time = time.time()
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.rank, end_time - start_time))