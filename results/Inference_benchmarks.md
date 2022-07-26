# Throughput Test of Inference Tasks

## GPT-J

- Test on 4 P3.2xlarge;
- Configure:
  - Input sequence length: 512
  - Generate sequence length: 32
  - Max batch size: 48;
- Tuning token level pipe micro batch size:

| Token micro-batch-size | Prompt time | Generate time | Overall time |
|------------------------|-------------|---------------|--------------|
| 48                     | 1.93 s      | 4.89 s        | 6.82 s       | 
| 24                     | 1.92 s      | 3.34 s        | 5.26 s       | 
| 12                     | 1.93 s      | 2.59 s        | 4.52 s       | 
| 6                      | 1.93 s      | 4.95 s        | 6.88 s       | 
| 3                      | 1.92 s      | 9.21 s        | 11.13 s      | 
| 1                      | 1.94 s      | 29.55s        | 31.48 s      | 


## GPT-66B

- Test on 1 P4d.24xlarge (8 40G A100)
- Configure
  - Input sequence length: 1024
  - Generate sequence length: 100
  - Max batch size: 20;
  - 8 layer per GPU;
- Tuning token level pipe micro batch size:


| Token micro-batch-size | Prompt time | Generate time | Overall time | FLOPS per GPU |
|------------------------|-------------|---------------|--------------|---------------|
| 20                     | 2.71 s      | 42.78 s       | 45.49 s      | 16.44 TFLOPS  |
| 10                     | 2.72 s      | 42.82 s       | 45.54 s      | 16.42 TFLOPS  |
| 4                      | 2.71 s      | 130.78 s      | 133.49 s     | 5.60 TFLOPS   |


## GPT-175B (Estimated run)

- Test 1/3 of the workflow on 1 P4d.24xlarge (8 40G A100); cannot get 3 machines at the same time.
- Configure
  - Input sequence length: 1024
  - Generate sequence length: 100
  - Max batch size: 20;
  - 4 layer per GPU;
- Tuning token level pipe micro batch size:


| Token micro-batch-size | Prompt time | Generate time  | Overall time | Estimate time on 3 nodes | FLOPS per GPU |
|------------------------|-------------|----------------|--------------|--------------------------|---------------|
| 20                     | 2.14 s      | 29.58 s        | 31.72 s      | 95.16 s                  | 6.94 TFLOPS   |
| 10                     | 2.14 s      | 25.89 s        | 28.03 s      | 84.09 s                  | 7.86 TFLOPS   |
| 4                      | 2.14 s      | 66.34 s        | 68.48  s     | 205.44 s                 | 3.22 TFLOPS   |
