# Result of GPT-3 XL  In FP16

For pipeline only, we have:

- Fp16 results (Updated 2022/04/20)

- A cluster of 8 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 xl scale and partition:

  - number of layer: 24 (3 on each node) 
  - model dim: 2048
  - number of head: 16 (original one is 24, but it is not dividable by 2048? Anyway the FLOP is the same.)
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 96
  - micro-batch dim: 1 
  - Storage of a micro-batch: 8 MB
  
### Result of 2022/04/20

- gpipe
- fp16.
- use micro-batch size: 1
- max batch size (due to DRAM limits): 96
- One pipline use 8 p3.2xlarge


| Network setting                     | Single Pipeline  | Sharded PS DP-2 | Central PS DP-2 | Sharded PS DP-4 | Central PS DP-4 | Sharded PS DP-8 | Central PS DP-8 |
|-------------------------------------|------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| default (about 0.1ms; up to 10Gbps) | 8.14 s           | 9.01 s          | 9.06 s          | 9.25 s          | 9.36 s          | 9.78 s          | 9.59 s          |
| delay 1ms  bandwidth 5Gbps          | 8.79 s           | 10.16 s         | 10.50 s         | 10.54 s         | 10.76 s         | 10.88 s         | 10.89 s         |
| delay 5ms  bandwidth 2Gbps          | 10.69 s          | 12.37 s         | 13.49 s         | 13.92 s         | 13.90 s         | 15.07 s         | 14.27 s         |
| delay 10ms  bandwidth 1Gbps         | 15.64 s          | 20.56 s         | 23.92 s         | 24.13 s         | 24.30 s         | 27.39 s         | 24.83 s         |
| delay 50ms  bandwidth 1Gbps         | 16.23 s          | 21.01 s         | 24.26 s         | 24.21 s         | 24.97 s         | 27.68 s         | 25.66 s         |

### Analysis of memory usage:

Assume 3 layer on each node, () includes the special cases of first/last stage:
- No-computation demanded DRAM:
  - Model:
    - fp32 copy of the model: 592 MB  (first stage: 818 MB)
    - fp32 copy of the gradient: 592 MB  (first stage: 818 MB)
    - fp16 copy of the model: 296 MB  (first stage: 409 MB)
    - fp16 copy of the gradient: 296 MB (first stage: 409 MB)
    - Total: 1.73 GB (first stage: 2.39 GB)
  - Activation/Gradient:
    - stage a_in buffer (receive activation from Rank i-1 node) in fp16: 8 MB X 96 = 768 MB (first stage does not have this.) 
    - stage d_in buffer (send activation gradient to Rank i-1 node) in fp16: 8 MB X 96 = 768 MB (first stage does not have this.) 
    - stage a_out buffer (send activation to Rank i+1 node)  in fp16: 8 MB X 96 = 768 MB (last stage does not have this.)
    - stage d_out buffer (receive activation gradient from Rank i+1 node)  in fp16: 8 MB X 96 = 768 MB (last stage does not have this.)
    - Total: 3 GB (first stage: 1.5 GB, last stage 1.5 GB)
  - Total:  4.73 GB (first stage: 3.89 GB, last stage: 3.23 GB)


### Result of 2022/04/23

Check the largest batch size for different number of layers: 

- Standard GPT-XL 
  - Include fp32 model offload.
  - Total of 8 P3.2xlarge instances, each hold 3 layers:

| Network setting                     | Batch-size 108 | Batch-size 96 | Batch-size 64 | Batch-size 32 |
|-------------------------------------|----------------|---------------|---------------|---------------|
| default (about 0.1ms; up to 10Gbps) | 9.30 s         | 8.34 s        | 5.97 s        | 3.51 s        |
| delay 1ms  bandwidth 5Gbps          | 10.20 s        | 9.36 s        | 6.20 s        | 3.63 s        |
| delay 5ms  bandwidth 2Gbps          | 11.60 s        | 11.02 s       | 8.31 s        | 5.09 s        |
| delay 10ms  bandwidth 1Gbps         | 17.78 s        | 16.27 s       | 12.04 s       | 8.50 s        |
| delay 50ms  bandwidth 1Gbps         | 18.41 s        | 16.74 s       | 12.62 s       | 10.06 s       |


- Scale with number of layers.
  - Fix a batch size of 64:

| Network setting                     | 3 Layer | 4 Layer | 5 Layer |
|-------------------------------------|---------|---------|---------|
| default (about 0.1ms; up to 10Gbps) | 5.97 s  | 7.30 s  | 8.86 s  |
| delay 1ms  bandwidth 5Gbps          | 6.20 s  | 7.62 s  | 9.33 s  |
| delay 5ms  bandwidth 2Gbps          | 8.31 s  | 10.06 s | 10.4 s  |
| delay 10ms  bandwidth 1Gbps         | 12.04 s | 12.16 s | 12.51 s |
| delay 50ms  bandwidth 1Gbps         | 12.62 s | 12.91 s | 13.19 s |


### Result of Megatron (2022/04/26)

- Megatron Baseline for GPT3-XL
  - Follow some setting from the Alpa Paper. 
  - Their setting: 64 V100 8 P3.16xlarge AWS instances.
  - Global batch size 1024, sequence length 2048 (This is different, Alpa use 1024).
  - They do not report micro-batch size, I use micro-batch size = 1.
  - Number of Para: 1.3B
  - Computation (in PFlop): 23.64
    - Forward: 5.91
    - Backward (with activation recompute X3): 17.73
  - DP degree: 8.

| Setting       | Tensor(T)-8 | Pipe(P)-8 | T-4 P-2 | T-2 P-4 |
|---------------|-------------|-----------|---------|---------|
| 8 P3.16xlarge |  s          |  s        |  s      |  s      |
| 64 P3.2xlarge |  s          |  s        |  s      |  s      |

