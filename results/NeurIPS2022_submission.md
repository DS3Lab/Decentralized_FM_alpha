# The Experiments for NeurIPS 2022 Submission

## Scenarios

- Case 1 data center on demand: 
  - 8 X p3.16xlarge (8 V100)
  - Intra-instance bandwidth 100 Gbps
  - Inter-instance bandwidth 25 Gbps
- Case 2 data center spot: 
  - 8 * p3.8xlarge(4 V100) + 32 p3.2xlarge (1 V100)
  - Intra-instance bandwidth 100 Gbps
  - Inter-instance bandwidth 10 Gbps
- Case 3 Multiple universities: 
  - 2 universities (each has 32 V100 GPUs)
  - With in each university bandwidth: 5 Gbps 
  - Connection between regions: 1.2 Gbps
- Case 4 Regional distributed: 
  - 4 regions in Europe 
  - Within each region: 2 Gbps 
  - across different region 8: 1.0 ~ 1.2 Gbps
- Case 5 World-wide distributed: 
  - 8 regions around the work 
  - Within each region: 2 Gbps
  - Across different region 0.3 ~ 1.2 Gbps


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

## Case 1 Data Center on Demand

- Notes:
  - Use aws_run_gpt3_Ngpu_training.sh for ours w scheduler; 
  - Use aws_run_gpt3_scheduled_Ngpu_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.
  
- One iteration runtime (in seconds):

| Setting   | Megatron-P8 | Megatron-T8 | Megatron-P4T2 | Megatron-P2T4 | Ours w scheduler | Ours wo Scheduler |
|-----------|-------------|-------------|---------------|---------------|------------------|-------------------|
| L24 B1024 | 24.34       | 25.04       | **12.52**     | 14.25         | **10.14**        | 20.27             |
| L24 B2048 | 50.73       | -           | **24.70**     | -             | **17.62**        | 33.25             |
| L24 B4096 | 101.79      | -           | **48.91**     | -             | **32.99**        | 65.78             |
| L32 B1024 | 26.80       | 32.59       | **15.38**     | 18.44         | **13.33**        | 23.09             |
| L32 B2048 | 53.45       | -           | **30.51**     | -             | **23.22**        | 39.43             |
| L32 B4096 | 106.82      | -           | **60.32**     | -             | **43.10**        | 78.47             |
| L40 B1024 | 28.62       | 40.46       | **18.88**     | 22.60         | **16.70**        | 24.48             |
| L40 B2048 | 56.78       | -           | **36.87**     | -             | **29.58**        | 44.71             |
| L40 B4096 | 111.58      | -           | **72.96**     | -             | **53.80**        | 88.74             |


- Hardware Efficiency (by PFlops):

| Setting   | Megatron-PP | Megatron-Opt | Ours w scheduler | Ours wo Scheduler |
|-----------|-------------|--------------|------------------|-------------------|
| L24 B1024 | 0.971       | 1.888        | 2.331            | 1.166             |
| L24 B2048 | 0.932       | 1.914        | 2.683            | 1.422             |
| L24 B4096 | 0.929       | 1.933        | 2.866            | 1.438             | 
| L32 B1024 | 1.176       | 2.049        | 2.365            | 1.365             |
| L32 B2048 | 1.179       | 2.066        | 2.715            | 1.599             |
| L32 B4096 | 1.180       | 2.090        | 2.925            | 1.607             |
| L40 B1024 | 1.377       | 2.087        | 2.360            | 1.610             |
| L40 B2048 | 1.388       | 2.138        | 2.665            | 1.763             |
| L40 B4096 | 1.413       | 2.161        | 2.930            | 1.776             |


## Case 2 Data Center Spot

- Notes:
  - Use aws_run_megatron_heter_gpu_training.sh to run Megatron 
  - Use aws_run_gpt3_Ngpu_training.sh for ours w scheduler; 
  - Use aws_run_gpt3_scheduled_Ngpu_training.sh for ours wo scheduler;

- One iteration runtime (in seconds):

| Setting   | Megatron-P8 | Megatron-T8 | Megatron-P4T2 | Megatron-P2T4 | Ours w scheduler | Ours wo Scheduler |
|-----------|-------------|-------------|---------------|---------------|------------------|-------------------|
| L24 B1024 | 21.31       | 265.42      | 48.48         | 118.99        | 12.33            | 15.30             |
| L24 B2048 | 42.37       | -           | -             | -             | 22.92            | 28.12             |
| L24 B4096 | 83.57       | -           | -             | -             | 45.24            | 53.72             |
| L32 B1024 | 23.18       | -           | -             | -             | 15.73            | 18.73             |
| L32 B2048 | 43.68       | -           | -             | -             | 28.97            | 33.48             |
| L32 B4096 | 85.54       | -           | -             | -             | 55.99            | 64.44             |
| L40 B1024 | 25.29       | -           | -             | -             | 18.43            | 21.67             |
| L40 B2048 | 49.41       | -           | -             | -             | 34.39            | 40.03             |
| L40 B4096 | 96.16       | -           | -             | -             | 67.23            | 77.20             |

- Hardware Efficiency (by PFlops):

| Setting    | Megatron-PP(Opt) | Ours w scheduler | Ours wo Scheduler |
|------------|------------------|------------------|-------------------|
| L24 B1028  | 1.109            | 1.917            | 1.545             |
| L24 B2048  | 1.116            | 2.063            | 1.681             |
| L24 B4096  | 1.132            | 2.090            | 1.760             |
| L32 B1024  | 1.360            | 2.004            | 1.683             |
| L32 B2048  | 1.443            | 2.176            | 1.883             |
| L32 B4096  | 1.474            | 2.252            | 1.957             |
| L40 B1024  | 1.558            | 2.138            | 1.819             |
| L40 B2048  | 1.595            | 2.292            | 1.969             |
| L40 B4096  | 1.639            | 2.345            | 2.042             |


## Case 3 Multiple Universities

- Notes:
  - Generate tc scripts: aws_generate_heter_tc.sh 2 (case indexed from 0)
  - Use aws_run_gpt3_1gpu_training.sh for ours w scheduler(manually optimized); 
  - Use aws_run_gpt3_scheduled_1gpu_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.

- One iteration runtime

| Setting   | Megatron-P8 (Default) | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|-----------|-----------------------|----------------------------|------------------|-------------------|
| L24 B1024 | 21.27                 | 33.72                      | 19.06            | 27.80             |
| L24 B2048 | 40.19                 | 59.69                      | 32.67            | 50.01             |
| L24 B4096 | 81.40                 | 111.31                     | 54.50            | 93.35             |
| L32 B1024 | 24.75                 | 33.98                      | 22.35            | 29.62             |
| L32 B2048 | 45.15                 | 60.17                      | 37.39            | 51.22             |
| L32 B4096 | 90.45                 | 112.40                     | 64.45            | 94.91             |
| L40 B1024 | 26.83                 | 37.01                      | 27.02            | 32.62             |
| L40 B2048 | 50.01                 | 65.06                      | 44.91            | 55.95             |
| L40 B4096 | 97.93                 | 121.98                     | 78.23            | 102.34            |

- Hardware Efficiency (by PFlops):

| Setting    | Megatron-PP(Default) | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|------------|----------------------|----------------------------|------------------|-------------------|
| L24 B1024  | 1.111                | 0.701                      | 1.240            | 0.850             |
| L24 B2048  | 1.176                | 0.792                      | 1.447            | 0.945             |
| L24 B4096  | 1.162                | 0.850                      | 1.735            | 1.013             |
| L32 B1024  | 1.274                | 0.928                      | 1.410            | 1.064             |
| L32 B2048  | 1.396                | 1.048                      | 1.686            | 1.231             |
| L32 B4096  | 1.394                | 1.122                      | 1.956            | 1.328             |
| L40 B1024  | 1.469                | 1.065                      | 1.459            | 1.208             |
| L40 B2048  | 1.576                | 1.211                      | 1.755            | 1.409             |
| L40 B4096  | 1.610                | 1.292                      | 2.015            | 1.540             |


## Case 4 Regional Distributed

- Notes:
  - Generate tc scripts: aws_generate_heter_tc.sh 3 (case indexed from 0)
  - Use aws_run_gpt3_1gpu_training.sh for ours w scheduler(manually optimized); 
  - Use aws_run_gpt3_scheduled_1gpu_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.

- One iteration runtime 

| Setting   | Megatron-P8 (Default)  | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|-----------|------------------------|----------------------------|------------------|-------------------|
| L24 B1024 | 38.77                  | 56.20                      | 22.24            | 31.99             |
| L24 B2048 | 72.91                  | 102.08                     | 38.68            | 57.07             |
| L24 B4096 | 141.71                 | 193.72                     | 72.02            | 106.12            |
| L32 B1024 | 42.59                  | 60.31                      | 26.24            | 33.78             |
| L32 B2048 | 79.96                  | 108.64                     | 45.66            | 58.42             |
| L32 B4096 | 151.56                 | 206.11                     | 86.07            | 108.91            |
| L40 B1024 | 46.63                  | 63.86                      | 31.69            | 37.62             |
| L40 B2048 | 85.65                  | 114.33                     | 51.75            | 63.58             |
| L40 B4096 | 164.77                 | 216.38                     | 95.46            | 115.90            |

- Hardware Efficiency (by PFlops):

| Setting   | Megatron-PP (Default) | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|-----------|-----------------------|----------------------------|------------------|-------------------|
| L24 B1024 | 0.610                 | 0.421                      | 1.063            | 0.739             |
| L24 B2048 | 0.648                 | 0.463                      | 1.222            | 0.828             |
| L24 B4096 | 0.667                 | 0.488                      | 1.313            | 0.891             |
| L32 B1024 | 0.740                 | 0.523                      | 1.201            | 0.933             |
| L32 B2048 | 0.788                 | 0.580                      | 1.381            | 1.079             |
| L32 B4096 | 0.832                 | 0.612                      | 1.465            | 1.158             |
| L40 B1024 | 0.845                 | 0.617                      | 1.244            | 1.048             |
| L40 B2048 | 0.920                 | 0.689                      | 1.523            | 1.240             |
| L40 B4096 | 0.957                 | 0.729                      | 1.651            | 1.360             |


## Case 5 World-wide Distributed 

- Notes:
  - Generate tc scripts: aws_generate_heter_tc.sh 4 (case indexed from 0)
  - Use aws_run_gpt3_1gpu_training.sh for ours w scheduler(manually optimized); 
  - Use aws_run_gpt3_scheduled_1gpu_training.sh for ours wo scheduler;
  - Use aws_run_megatron_training.sh for different Megatron Settings.

- One iteration runtime 

| Setting   | Megatron-P8 (Default) | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|-----------|-----------------------|----------------------------|------------------|-------------------|
| L24 B1024 | 83.04                 | 167.74                     | 43.78            | 92.29             |
| L24 B2048 | 156.93                | 301.30                     | 62.27            | 165.89            |
| L24 B4096 | 307.79                | 572.48                     | 104.06           | 313.63            |
| L32 B1024 | 85.97                 | 180.11                     | 51.06            | 99.10             |
| L32 B2048 | 165.14                | 313.90                     | 71.96            | 172.54            |
| L32 B4096 | 327.65                | 584.52                     | 114.01           | 320.63            |
| L40 B1024 | 89.35                 | 188.17                     | 58.24            | 106.71            |
| L40 B2048 | 171.95                | 326.02                     | 80.68            | 181.16            |
| L40 B4096 | 337.69                | 595.64                     | 124.07           | 331.25            |

- Hardware Efficiency (by PFlops):

| Setting    | Megatron-PP(Default) | Megatron-P8 (wo Scheduler) | Ours w scheduler | Ours wo Scheduler |
|------------|----------------------|----------------------------|------------------|-------------------|
| L24 B1024  | 0.285                | 0.141                      | 0.540            | 0.256             |
| L24 B2048  | 0.301                | 0.157                      | 0.759            | 0.285             |
| L24 B4096  | 0.307                | 0.165                      | 0.909            | 0.302             |
| L32 B1024  | 0.367                | 0.175                      | 0.617            | 0.318             |
| L32 B2048  | 0.382                | 0.201                      | 0.876            | 0.365             |
| L32 B4096  | 0.385                | 0.216                      | 1.106            | 0.393             |
| L40 B1024  | 0.441                | 0.209                      | 0.677            | 0.369             |
| L40 B2048  | 0.458                | 0.242                      | 0.977            | 0.435             |
| L40 B4096  | 0.467                | 0.265                      | 1.271            | 0.476             |