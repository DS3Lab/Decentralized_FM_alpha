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

- My command:

       python dist_pipeline_runner.py --dist-url tcp://172.31.36.218:9000 --world-size 3 --batch-size 64 --profiling no-profiling --num-iters 10 --embedding-dim 768 --num-layers 4 --rank 0 --mode gpipe --micro-batch-size 4

### Gpipe based pipeline parallel
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|--------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.64 s             | 6.76 s              | 6.94 s              |
| delay 1ms  bandwidth 5Gbps          | 6.68 s             | 6.82 s              | 7.01 s              |
| delay 5ms  bandwidth 2Gbps          | 6.70 s             | 6.96 s              | 7.31 s              |
| delay 10ms  bandwidth 1Gbps         | 8.38 s             | 8.61 s              | 9.08 s              |

### 1F1B based pipeline parallel
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.54 s              | 6.65 s              | 6.86 s              |
| delay 1ms  bandwidth 5Gbps          | 6.86 s              | 7.05 s              | 7.22 s              |
| delay 5ms  bandwidth 2Gbps          | 8.42 s              | 8.40 s              | 8.57 s              |
| delay 10ms  bandwidth 1Gbps         | 11.21 s             | 11.03 s             | 11.35 s             |

### ZeRO-S3 data parallel

- Fail due to running out of DRAM: 
  - for this model each data parallel node can only run a batch size of 1.
