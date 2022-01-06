## Result GPT-3 Small  

- A cluster of 3 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 small scale and partition:

  - number of layer: 12 (4 on each node) 
  - model dim: 768
  - number of head: 12
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 64
  - based on the batch size of 64, micro-batch size of 8 will break the DRAM limit. 


| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 7.17 s              | 7.04 s              | 7.08 s              |
| delay 1ms                           | 7.34 s              | 7.24 s              | 7.32 s              |
| delay 5ms                           | 7.62 s              | 7.83 s              | 8.15 s              |
| delay 10ms                          | 8.48 s              | 8.73 s              | 9.02 s              |
| bandwidth 5Gbps                     | 7.28 s              | 7.16 s              | 7.31 s              |
| bandwidth 2Gbps                     | 7.42 s              | 7.69 s              | 8.24 s              |
| bandwidth 1Gbps                     | 8.93 s              | 9.35 s              | 9.55 s              |
| delay 1ms  bandwidth 5Gbps          | 7.39 s              | 7.27 s              | 7.49 s              |
| delay 5ms  bandwidth 2Gbps          | 7.85 s              | 8.03 s              | 8.47 s              |
| delay 10ms  bandwidth 1Gbps         | 10.07 s             | 10.10 s             | 11.37 s             |
