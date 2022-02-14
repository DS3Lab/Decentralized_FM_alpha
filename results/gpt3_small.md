# Result of GPT-3 Small  

## Pipeline

For pipeline only, we have:

- A cluster of 3 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 small scale and partition:

  - number of layer: 12 (4 on each node) 
  - model dim: 768
  - number of head: 12
  - sequence length: 2048;
  - max batch size (due to DRAM limits): 64
  - micro-batch size: 1
  - Storage of parameters in an instances:  

### Gpipe based pipeline parallel 

- (updated on 2022/02/14).

| Network setting                     | Micro batch size: 1 | 
|-------------------------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 5.91 s              |
| delay 1ms  bandwidth 5Gbps          | 6.01 s              | 
| delay 5ms  bandwidth 2Gbps          | 6.07 s              | 
| delay 10ms  bandwidth 1Gbps         | 7.90 s              | 

### 1F1B based pipeline parallel 
- Not fast enough, left for further optimization.

| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.54 s              | 6.65 s              | 6.86 s              |
| delay 1ms  bandwidth 5Gbps          | 6.86 s              | 7.05 s              | 7.22 s              |
| delay 5ms  bandwidth 2Gbps          | 8.42 s              | 8.40 s              | 8.57 s              |
| delay 10ms  bandwidth 1Gbps         | 11.21 s             | 11.03 s             | 11.35 s             |






## ZeRO-S3 data parallel

- Fail due to running out of DRAM: 
  - for this model each data parallel node can only run a batch size of 1.
