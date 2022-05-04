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
- Switching batch sizes: 1024, 2048, 4096
- Sequence length: 2048
- Micro-batch size: 1
- fp16
- The PFlop for each setting:

|            | Layer 24 | Layer 32 | Layer 40 |
|------------|----------|----------|----------|
| Batch 1024 | 23.64    | 31.52    | 39.41    |
| Batch 2048 | 47.28    | 63.04    | 78.82    |
| Batch 4096 | 94.56    | 126.08   | 157.64   |

## Case 1 

- One iteration runtime (in seconds)
- Notes:
  - Use aws_run_gpt3_training.sh for ours w scheduler; 
  - Use aws_run_scheduled_gpt3_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.

| Setting   | Megatron-PP (P8T1) | Megatron-T8 | Megatron-P4T2 | Megatron-P2T4 | Ours w scheduler | Ours wo Scheduler |
|-----------|--------------------|-------------|---------------|---------------|------------------|-------------------|
| L24 B1024 | 24.34              | 25.04       | **12.52**     | 14.25         | **10.14**        | 20.27             |
| L24 B2048 | 50.73              | -           | **24.70**     | -             | **17.62**        | 33.25             |
| L24 B4096 | 101.79             | -           | **48.91**     | -             | **32.99**        | 65.78             |
| L32 B1024 | 26.80              | 32.59       | **15.38**     | 18.44         | **13.33**        | 23.09             |
| L32 B2048 | 53.45              | -           | **30.51**     | -             | **23.22**        | 39.43             |
| L32 B4096 | 106.82             | -           | **60.32**     | -             | **43.10**        | 78.47             |
| L40 B1024 | 28.62              | 40.46       | **18.88**     | 22.60         | **16.70**        | 24.48             |
| L40 B2048 | 56.78              | -           | **36.87**     | -             | **29.58**        | 44.71             |
| L40 B4096 | 111.58             | -           | **72.96**     | -             | **53.80**        | 88.74             |


- Hardware Efficiency (by PFlops)

| Setting   | Megatron-PP (P8T1) | Megatron-P4T2 | Ours w scheduler | Ours wo Scheduler |
|-----------|--------------------|---------------|------------------|-------------------|
| L24 B1024 | 0.971              | 1.888         | 2.331            | 1.166             |
| L24 B2048 | 0.932              | 1.914         | 2.683            | 1.422             |
| L24 B4096 | 0.929              | 1.933         | 2.866            | 1.438             | 
| L32 B1024 | 1.176              | 2.049         | 2.365            | 1.365             |
| L32 B2048 | 1.179              | 2.066         | 2.715            | 1.599             |
| L32 B4096 | 1.180              | 2.090         | 2.925            | 1.607             |
| L40 B1024 | 1.377              | 2.087         | 2.360            | 1.610             |
| L40 B2048 | 1.388              | 2.138         | 2.665            | 1.763             |
| L40 B4096 | 1.413              | 2.161         | 2.930            | 1.776             |



