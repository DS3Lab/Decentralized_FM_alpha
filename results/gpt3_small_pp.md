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

### Gpipe based pipeline parallel
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.71 s              | 6.85 s              | 7.04 s              |
| delay 1ms                           | -                   | -                   | -                   |
| delay 5ms                           | -                   | -                   | -                   |
| delay 10ms                          | -                   | -                   | -                   |
| bandwidth 5Gbps                     | -                   | -                   | -                   |
| bandwidth 2Gbps                     | -                   | -                   | -                   |
| bandwidth 1Gbps                     | -                   | -                   | -                   |
| delay 1ms  bandwidth 5Gbps          | -                   | -                   | -                   |
| delay 5ms  bandwidth 2Gbps          | -                   | -                   | -                   |
| delay 10ms  bandwidth 1Gbps         | 8.52 s              | 8.75 s              | 9.37 s              |

### 1F1B based pipeline parallel
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.68 s              | 6.82 s              | 7.11 s              |
| delay 1ms                           | -                   | -                   | -                   |
| delay 5ms                           | -                   | -                   | -                   |
| delay 10ms                          | -                   | -                   | -                   |
| bandwidth 5Gbps                     | -                   | -                   | -                   |
| bandwidth 2Gbps                     | -                   | -                   | -                   |
| bandwidth 1Gbps                     | -                   | -                   | -                   |
| delay 1ms  bandwidth 5Gbps          | -                   | -                   | -                   |
| delay 5ms  bandwidth 2Gbps          | -                   | -                   | -                   |
| delay 10ms  bandwidth 1Gbps         | 11.40 s             | 11.54 s             | 11.63 s             |

###ZeRO-S3 data parallel

- Fail due to running out of DRAM: 
  - for this model each data parallel node can only run a batch size of 1.
