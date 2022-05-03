# The Experiments for NeurIPS 2022 Submission

## Scenarios

- Case 1 data center on demand: 8 X p3.16xlarge(8 V100)
- Case 2 data center spot: 8 * p3.8xlarge(4 V100) + 32 p3.2xlarge
- Case 3 

## Compare Group

Four settings:
- (a). Megatron-Optimal 
- (b). Megatron-PP
- (c). Ours with Scheduler
- (d). Ours without Scheduler

Report:
- End-to-end: (a) vs (c) all scenarios 
- Effectiveness of scheduler: (c) vs (d) vs (b) all scenarios
- Effectiveness of system implementation: (b) vs (d) in homogeneous TC

Shared settings (GPT-XL):
- Switching total layers: 24, 32, 40
- Switching batch sizes: 1024, 1536, 2048
- Sequence length: 2048
- Micro-batch size: 1
- fp16


## Case 1 

- One iteration runtime (in seconds)
- Notes:
  - Use aws_run_gpt3_training.sh for ours w scheduler; 
  - Use aws_run_scheduled_gpt3_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.

| Setting   | Megatron-PP (P8T1) | Megatron-T8 | Megatron-P4T2 | Megatron-P2T4 | Ours w scheduler | Ours wo Scheduler |
|-----------|--------------------|-------------|---------------|---------------|------------------|-------------------|
| L24 B1024 | 24.34              | 25.04       | **12.52**     | 14.25         |                  |                   |
| L24 B2048 | 50.73              | -           | **24.70**     | -             |                  |                   |
| L24 B4096 | 101.79             | -           | **48.91**     | -             |                  |                   |
| L32 B1024 | 26.80              | 32.59       | **15.38**     | 18.44         |                  |                   |
| L32 B2048 | 53.45              | -           | **30.51**     | -             |                  |                   |
| L32 B4096 | 106.82             | -           | **60.32**     | -             |                  |                   |
| L40 B1024 | 28.62              | 40.46       | **18.88**     | 22.60         |                  |                   |
| L40 B2048 | 56.78              | -           | **36.87**     | -             |                  |                   |
| L40 B4096 | 111.58             | -           | **72.96**     | -             |                  |                   |







