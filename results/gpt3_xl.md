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
| delay 50ms  bandwidth 1Gbps         | 29.08 s              | 


## Pipeline + Data Parallel

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