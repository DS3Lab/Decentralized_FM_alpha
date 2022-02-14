## Result GPT-3 XL  

- A cluster of 12 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 xl scale and partition:

  - number of layer: 24 (2 on each node) 
  - model dim: 2048
  - number of head: 16 (original one is 24, but it is not dividable by 2048? Anyway the FLOP is the same.)
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 48
  - based on the batch size of 48, micro-batch size of 4 will break the DRAM limit. 

- My command:

       python dist_pipeline_runner.py --dist-url tcp://172.31.36.218:9000 --world-size 12 --batch-size 48 --profiling no-profiling --num-iters 10 --embedding-dim 2048 --num-layers 2 --rank 0 --mode gpipe --micro-batch-size 3

### Gpipe based pipeline parallel
| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 3 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) |  s             |  s             |  s             |
| delay 1ms                           | -                   | -                   | -                   |
| delay 5ms                           | -                   | -                   | -                   |
| delay 10ms                          | -                   | -                   | -                   |
| bandwidth 5Gbps                     | -                   | -                   | -                   |
| bandwidth 2Gbps                     | -                   | -                   | -                   |
| bandwidth 1Gbps                     | -                   | -                   | -                   |
| delay 1ms  bandwidth 5Gbps          | -                   | -                   | -                   |
| delay 5ms  bandwidth 2Gbps          | -                   | -                   | -                   |
| delay 10ms  bandwidth 1Gbps         |  s             |  s             |  s             |


### 1F1B based pipeline parallel
