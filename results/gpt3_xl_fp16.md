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

- Run time profiling:
  - Local simulation without communication: ()[]
  - Real execution: 



