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


| Network setting                     | DP-1    | Sharded PS DP-2 | Central PS DP-2 | Sharded PS DP-4 | Central PS DP-4 |
|-------------------------------------|---------|-----------------|-----------------|-----------------|-----------------|
| default (about 0.1ms; up to 10Gbps) | 8.14 s  | 9.01 s          | 9.06 s          |                 |                 |
| delay 1ms  bandwidth 5Gbps          | 8.79 s  | 10.16 s         | 10.50 s         |                 |                 |
| delay 5ms  bandwidth 2Gbps          | 10.69 s | 12.37 s         | 13.49 s         |                 |                 |
| delay 10ms  bandwidth 1Gbps         | 15.64 s | 20.56 s         | 23.92 s         |                 |                 |
| delay 50ms  bandwidth 1Gbps         | 16.23 s | 21.01 s         | 24.26 s         |                 |                 |