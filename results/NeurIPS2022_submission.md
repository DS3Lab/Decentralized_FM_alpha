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
| L24 B1024 | 16.91              | 25.04       | **12.52**     | 14.25         |                  |                   |
| L24 B2048 |                    |             |               |               |                  |                   |
| L24 B4096 |                    |             |               |               |                  |                   |
| L32 B1024 |                    |             |               |               |                  |                   |
| L32 B2048 |                    |             |               |               |                  |                   |
| L32 B4096 |                    |             |               |               |                  |                   |
| L40 B1024 |                    |             |               |               |                  |                   |
| L40 B2048 |                    |             |               |               |                  |                   |
| L40 B4096 |                    |             |               |               |                  |                   |







