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

### Gpipe based pipeline
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | s | s | s                   |
| delay 1ms                           | s | s | s                   |
| delay 5ms                           | s | s | s                   |
| delay 10ms                          | s | s | s                   |
| bandwidth 5Gbps                     | s | s | s                   |
| bandwidth 2Gbps                     | s | s | s                   |
| bandwidth 1Gbps                     | s | s | s                   |
| delay 1ms  bandwidth 5Gbps          | s | s | s                   |
| delay 5ms  bandwidth 2Gbps          | s | s | s                   |
| delay 10ms  bandwidth 1Gbps         | s | s | 8.97 s              |
### 1F1B based pipeline
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---|---|---------------------|
| default (about 0.1ms; up to 10Gbps) | s | s | s                   |
| delay 1ms                           | s | s | s                   |
| delay 5ms                           | s | s | s                   |
| delay 10ms                          | s | s | s                   |
| bandwidth 5Gbps                     | s | s | s                   |
| bandwidth 2Gbps                     | s | s | s                   |
| bandwidth 1Gbps                     | s | s | s                   |
| delay 1ms  bandwidth 5Gbps          | s | s | s                   |
| delay 5ms  bandwidth 2Gbps          | s | s | s                   |
| delay 10ms  bandwidth 1Gbps         | s | s | 11.s                |
