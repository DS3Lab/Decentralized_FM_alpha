# Result of GPT-3 XL  

## Pipeline Alone

For pipeline only, we have:

- A cluster of 12 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 xl scale and partition:

  - number of layer: 24 (2 on each node) 
  - model dim: 2048
  - number of head: 16 (original one is 24, but it is not dividable by 2048? Anyway the FLOP is the same.)
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 64
  - micro-batch dim: 1 
  - Storage of a micro-batch: 16 MB
  - Storage of parameters in a regular instance: 400 MB 
  - Storage of parameters in the first instance of pipeline (due to some NLP embedding): 626 MB
  
### Gpipe based pipeline parallel

- (updated on 2022/02/14).

| Network setting                     | Micro batch size: 1 | 
|-------------------------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 15.16 s             |
| delay 1ms  bandwidth 5Gbps          | 15.49 s             | 
| delay 5ms  bandwidth 2Gbps          | 16.80 s             | 
| delay 10ms  bandwidth 1Gbps         | 22.90 s             |
| delay 50ms  bandwidth 1Gbps         | 29.08 s             | 


### PyTorch Pipe Baseline

- Runs on a P3.16xlarge machine with 8 V100 each (16GB dRAM)
- To run a batch size of 64:

      python dist_torch_pipe.py --cuda-num 8 --num-layers 24 --embedding-dim 2048 --batch-size 64 --micro-batch-size 1/2 

| Micro batch size | execution time |
|------------------|----------------|
| 1                | 17.78 s        |
| 2                | 18.77 s        |
| 4                | Fail           |

When micro-batch size is larger than 4, it would fail due to OOM. 


### PyTorch Pipe + Fairscale FSDP

- Runs on 6 P3.16xlarge machine with 8 V100 each (16GB dRAM)
- To run a batch size of 252 (notice --world-size is 6 instead of 48):
  
      python dist_fairscale_pipe_fsdp.py --world-size 6 --dist-backend nccl --dist-url tcp://172.31.14.156:6000 --cuda-num 8 --num-layers 24 --seq-length 2048 --embedding-dim 2048 --batch-size 42 --micro-batch-size 1 --rank 0
     
| Micro batch size | execution time |
|------------------|-----------|
| 1                |  s        |
| 2                |  s        |
| 4                | Fail      |



### Megatron Tensor/Pipeline/Data Baseline

- Runs on a P3.16xlarge machine with 8 V100 each (16GB dRAM)
- fp32 & enabled checkpointing (activation recompute)
- To run a batch size of 64:
  
       sh ./scripts/local_scripts/local_test_multi_GPU_megatron_QQP.sh $MICRO_BATCH_SIZE $PIPELINE_PARALLEL_SIZE $TENSOR_PARALLEL_SIZE 0

| Micro batch size | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4 |
|------------------|-------------|-----------|---------|---------|
| 1                | 25.00 s     | 19.71 s   | 20.64 s | 20.81 s |
| 2                | 24.18 s     | 21.52 s   | 26.72 s | 21.46 s |
| 4                | 23.31 s     | 25.14 s   | 20.91 s | 22.79 s | 
| 8                | 20.90 s     | Fail      | 21.67 s | Fail    | 
| 16               | Fail        | Fail      | Fail    | Fail    |

- To run a batch size of 64:
- fp16 & enabled checkpointing (activation recompute)
- (add -fp16 in the script)

| Micro batch size | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4 |
|------------------|-------------|-----------|---------|---------|
| 1                | s           | s         | s       | s       |
| 2                | s           | s         | s       | s       |
| 4                | s           | s         | s       | s       | 
| 8                | s           | s         | s       | s       | 
| 16               | s           | s         | s       | s       | 


- To run a batch size of 252:
- 6 p3.16xlarge instances (25 Gbps inter-node connection)
- DP degree 6

- FP32 & activation recompute:

| Micro batch size | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4 |
|------------------|-------------|-----------|---------|---------|
| 1                | 19.71 s     | 20.04 s   | 15.91 s | 25.59 s |
| 2                | 20.46 s     | 18.90 s   | 17.79 s | 21.34 s |
| 3                | 23.48 s     | 19.44 s   | 16.18 s | 20.59 s |

- FP16 & activation recompute:

| Micro batch size | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4 |
|------------------|-------------|-----------|---------|---------|
| 1                | 9.19 s      | 7.58 s    | 5.25 s  | 10.86 s |
| 2                | 41.06 s ??  | 7.47 s    | 12.45 s | 10.33 s |
| 3                | 20.68 s     | 6.07 s    | 7.33 s  | 8.68 s  |

- FP16 & NO activation recompute:

| Micro batch size | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4    |
|------------------|-------------|-----------|---------|------------|
| 1                | 7.22 s      | 7.03 s    | 4.21 s  | 10.09 s    |
| 2                | 18.75 s ??  | Fail(OOM) | 6.78 s  | 9.06 s     |
| 3                | 10.18 s     | Fail(OOM) | 4.51 s  | Fail(OOM)  |

(Not sure what happens for config ?? Run this twice with the same result.)


## Ours Pipeline + Data Parallel

- Gpipe + centralized PS Data parallel:
- Each GPipe pipeline runs a batch of size 64;
- The global batch size is 64 * DP degree:
  - DP degree 1: 64
  - DP degree 4: 256
- (updated on 2022/02/14).

| Network setting                     | DP Degree: 1 | DP Degree: 4 | 
|-------------------------------------|--------------|--------------|
| default (about 0.1ms; up to 10Gbps) | 15.16 s      | 16.17 s      |
| delay 1ms  bandwidth 5Gbps          | 15.49 s      | 17.55 s      |
| delay 5ms  bandwidth 2Gbps          | 16.80 s      | 21.99 s      |
| delay 10ms  bandwidth 1Gbps         | 22.90 s      | 33.39 s      |