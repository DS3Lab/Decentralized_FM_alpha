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


## OPT-66

- Test on 1 P4d.24xlarge (8 40G A100)
- Configure
  - Input sequence length: 1024
  - Generate sequence length: 100
  - Max batch size: 20;
- Tuning token level pipe micro batch size:


| Token micro-batch-size | Prompt time | Generate time | Overall time |
|------------------------|-------------|---------------|--------------|
| 20                     | 2.71 s      | 42.78 s       | 45.49 s      | 
| 10                     | 2.72 s      | 42.82 s       | 45.54 s      | 
| 4                      | 2.71 s      | 130.78 s      | 133.49 s     | 
| 2                      | s           | s             | s            | 
| 1                      | s           | s             | s            |