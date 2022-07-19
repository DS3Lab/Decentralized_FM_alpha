# Throughput Test of Inference Tasks

## GPT-J

- Test on 4 P3.2xlarge;
- Configure:
  - Input sequence length: 512
  - Generate sequence length: 32
  - Max batch size: 48;
- Tuning Token level pipe micro batch size:

| Token micro-batch-size | Prompt time | Generate time | Overall time |
|------------------------|-------------|---------------|--------------|
| 48                     | 1.93 s      | 4.89 s        | 6.82 s       | 
| 24                     | 1.92 s      | 3.34 s        | 5.26 s       | 
| 12                     | 1.93 s      | 2.59 s        | 4.52 s       | 
| 6                      | 1.93 s      | 4.95 s        | 6.88 s       | 
| 3                      | 1.92 s      | 9.21 s        | 11.13 s      | 
| 1                      | 1.94 s      | 29.55s        | 31.48 s      | 


## OPT-66