## Result GPT-3 XL  

- A cluster of 12 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 xl scale and partition:

  - number of layer: 12 (4 on each node) 
  - model dim: 2048
  - number of head: 16 (original one is 24, but it is not dividable by 2048? Anyway the FLOP is the same.)
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 48
  - based on the batch size of 48, micro-batch size of 4 will break the DRAM limit. 


| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 3 |
|-------------------------------------|---------------------|---------------------|-----------------|
| default (about 0.1ms; up to 10Gbps) | 11.87 s             | 14.08 s             | 15.62 s         |
| delay 1ms                           | 12.18 s             | 14.32 s             | 16.12 s         |
| delay 5ms                           | 13.04 s             | 15.06 s             | 16.94 s         |
| delay 10ms                          | 14.28 s             | 16.53 s             | 18.33 s         |
| bandwidth 5Gbps                     | 12.26 s             | 14.99 s             | 17.04 s         |
| bandwidth 2Gbps                     | 13.61 s             | 17.18 s             | 20.78 s         |
| bandwidth 1Gbps                     | 18.51 s             | 24.36 s             | 29.71 s         |
| delay 1ms  bandwidth 5Gbps          | 12.34 s             | 15.08 s             | 17.21 s         |
| delay 5ms  bandwidth 2Gbps          | 13.91 s             | 17.36 s             | 21.47 s         |
| delay 10ms  bandwidth 1Gbps         | 19.46 s             | 24.74 s             | 30.84 s         |
