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

| Setting   | Megatron-P8 | Megatron-T8 | Megatron-P2T4 | Megatron-P4T2 |        |        | Ours w scheduler |        |        | Ours wo Scheduler |        |        |
|-----------|-------------|-------------|---------------|---------------|--------|--------|------------------|--------|--------|-------------------|--------|--------|
| Seed      | -           | -           | -             | 2022          | 2023   | 2024   | 2022             | 2023   | 2024   | 2022              | 2023   | 2024   |
| L24 B1024 | 24.34       | 25.04       | 14.25         | 12.52         | 12.55  | 12.54  | 10.14            | 10.21  | 10.08  | 20.27             | 19.46  | 20.19  |
| L24 B2048 | 50.73       | -           | -             | 24.70         | 24.71  | 24.66  | 17.62            | 17.63  | 17.81  | 33.25             | 32.32  | 33.10  |
| L24 B4096 | 101.79      | -           | -             | 48.91         | 48.76  | 48.85  | 32.99            | 32.98  | 33.19  | 65.78             | 67.13  | 66.43  |
| L32 B1024 | 26.80       | 32.59       | 18.44         | 15.38         | 15.43  | 15.30  | 13.33            | 13.24  | 13.13  | 23.09             | 24.72  | 22.12  |
| L32 B2048 | 53.45       | -           | -             | 30.51         | 30.31  | 30.62  | 23.22            | 23.21  | 23.12  | 39.43             | 40.25  | 39.29  |
| L32 B4096 | 106.82      | -           | -             | 60.32         | 60.39  | 60.31  | 43.10            | 43.34  | 43.18  | 78.47             | 79.06  | 79.59  |
| L40 B1024 | 28.62       | 40.46       | 22.60         | 18.88         | 18.77  | 18.60  | 16.70            | 16.68  | 16.65  | 24.48             | 25.51  | 23.04  |
| L40 B2048 | 56.78       | -           | -             | 36.87         | 36.89  | 36.86  | 29.58            | 29.51  | 29.42  | 44.71             | 44.44  | 43.80  |
| L40 B4096 | 111.58      | -           | -             | 72.96         | 72.93  | 72.88  | 53.80            | 53.76  | 53.84  | 88.74             | 88.71  | 88.60  |


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

| Setting   | Megatron-T8 | Megatron-P4T2 | Megatron-P2T4 | Megatron-P8 |       |        | Ours w scheduler |        |        | Ours wo Scheduler |       |        |
|-----------|-------------|---------------|---------------|-------------|-------|--------|------------------|--------|--------|-------------------|-------|--------|
| Seed      | -           | -             | -             |  2022       | 2023  | 2024   | 2022             | 2023   | 2024   | 2022              | 2023  | 2024   |
| L24 B1024 | 265.42      | 48.48         | 118.99        | 21.31       | 21.79 | 20.79  | 12.33            | 12.35  | 12.38  | 15.30             | 14.99 | 14.96  | 
| L24 B2048 | -           | -             | -             | 42.37       | 42.04 | 42.31  | 22.92            | 23.07  | 23.01  | 28.12             | 27.83 | 27.99  | 
| L24 B4096 | -           | -             | -             | 83.57       | 83.05 | 83.93  | 45.24            | 45.17  | 45.59  | 53.72             | 53.56 | 52.47  | 
| L32 B1024 | -           | -             | -             | 23.18       | 22.94 | 23.91  | 15.73            | 15.27  | 15.83  | 18.73             | 18.89 | 18.99  | 
| L32 B2048 | -           | -             | -             | 43.68       | 44.45 | 43.57  | 28.97            | 28.98  | 28.70  | 33.48             | 32.23 | 34.20  | 
| L32 B4096 | -           | -             | -             | 85.54       | 85.34 | 85.08  | 55.99            | 55.64  | 55.88  | 64.44             | 66.42 | 63.78  | 
| L40 B1024 | -           | -             | -             | 25.29       | 25.34 | 25.39  | 18.43            | 18.60  | 18.44  | 21.67             | 22.61 | 20.74  | 
| L40 B2048 | -           | -             | -             | 49.41       | 49.53 | 49.36  | 34.39            | 34.91  | 33.42  | 40.03             | 41.67 | 40.30  | 
| L40 B4096 | -           | -             | -             | 96.16       | 96.06 | 95.68  | 67.23            | 66.90  | 68.23  | 77.20             | 79.32 | 77.23  | 

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

| Setting   | Megatron-P8 (Default) | Megatron-P8 (wo Scheduler) |         |         | Ours w scheduler |       |        | Ours wo Scheduler |       |       |
|-----------|-----------------------|----------------------------|---------|---------|------------------|-------|--------|-------------------|-------|-------|
| Seed      | -                     | 2022                       | 2023    | 2024    | 2022             | 2023  | 2024   | 2022              | 2023  | 2024  |
| L24 B1024 | 21.27                 | 33.72                      | 32.59   | 32.77   | 19.06            | 18.92 | 18.63  | 27.80             | 27.70 | 27.96 |
| L24 B2048 | 40.19                 | 59.69                      | 58.43   | 59.01   | 32.67            | 32.47 | 32.20  | 50.01             | 49.82 | 50.19 |
| L24 B4096 | 81.40                 | 111.31                     | 110.31  | 110.49  | 54.50            | 54.38 | 53.86  | 93.35             | 92.33 | 92.91 |
| L32 B1024 | 24.75                 | 33.98                      | 33.02   | 33.17   | 22.35            | 22.03 | 21.95  | 29.62             | 29.01 | 29.57 |
| L32 B2048 | 45.15                 | 60.17                      | 59.21   | 59.48   | 37.39            | 36.92 | 37.24  | 51.22             | 50.93 | 50.86 |
| L32 B4096 | 90.45                 | 112.40                     | 111.39  | 111.66  | 64.45            | 64.29 | 63.92  | 94.91             | 94.07 | 94.32 |
| L40 B1024 | 26.83                 | 37.01                      | 35.97   | 36.01   | 27.02            | 26.94 | 26.38  | 32.62             | 31.05 | 31.76 |
| L40 B2048 | 50.01                 | 65.06                      | 64.05   | 64.12   | 44.91            | 44.81 | 44.38  | 55.95             | 55.21 | 55.61 |
| L40 B4096 | 97.93                 | 121.98                     | 121.10  | 121.02  | 78.23            | 78.87 | 78.79  | 102.34            | 98.83 | 99.29 |

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

| Setting   | Megatron-P8 (Default)  | Megatron-P8 (wo Scheduler) |         |         | Ours w scheduler |        |        | Ours wo Scheduler |        |        |
|-----------|------------------------|----------------------------|---------|---------|------------------|--------|--------|-------------------|--------|--------|
| Seed      | -                      | 2022                       | 2023    | 2024    | 2022             | 2023   | 2024   | 2022              | 2023   | 2024   |
| L24 B1024 | 38.77                  | 56.20                      | 62.36   | 59.76   | 22.24            | 21.85  | 21.81  | 31.99             | 31.77  | 31.96  |
| L24 B2048 | 72.91                  | 102.08                     | 106.41  | 104.93  | 38.68            | 38.36  | 38.41  | 57.07             | 56.98  | 57.01  |
| L24 B4096 | 141.71                 | 193.72                     | 197.92  | 195.62  | 72.02            | 71.76  | 71.76  | 106.12            | 105.76 | 106.02 |
| L32 B1024 | 42.59                  | 60.31                      | 64.82   | 63.51   | 26.24            | 26.01  | 25.97  | 33.78             | 33.62  | 33.59  |
| L32 B2048 | 79.96                  | 108.64                     | 112.52  | 111.90  | 45.66            | 45.76  | 45.38  | 58.42             | 58.54  | 58.60  |
| L32 B4096 | 151.56                 | 206.11                     | 211.91  | 209.05  | 86.07            | 85.73  | 85.79  | 108.91            | 107.81 | 108.21 |
| L40 B1024 | 46.63                  | 63.86                      | 69.48   | 66.63   | 31.69            | 31.43  | 31.38  | 37.62             | 37.43  | 37.39  |
| L40 B2048 | 85.65                  | 114.33                     | 120.37  | 117.03  | 51.75            | 51.55  | 51.45  | 63.58             | 62.12  | 62.48  |
| L40 B4096 | 164.77                 | 216.38                     | 221.44  | 218.43  | 95.46            | 95.15  | 95.11  | 115.90            | 112.34 | 113.74 |

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

| Setting   | Megatron-P8 (Default) | Megatron-P8 (wo Scheduler) |         |         | Ours w scheduler |         |         | Ours wo Scheduler |        |         |
|-----------|-----------------------|----------------------------|---------|---------|------------------|---------|---------|-------------------|--------|---------|
| Seed      | -                     | 2022                       | 2023    | 2024    | 2022             | 2023    | 2024    | 2022              | 2023   | 2024    |
| L24 B1024 | 83.04                 | 167.74                     | 164.38  | 164.63  | 43.78            | 44.39   | 43.87   | 92.29             | 89.64  | 93.46   |
| L24 B2048 | 156.93                | 301.30                     | 297.88  | 298.45  | 62.27            | 62.90   | 62.59   | 165.89            | 162.10 | 163.80  |
| L24 B4096 | 307.79                | 572.48                     | 569.22  | 569.25  | 104.06           | 104.53  | 104.08  | 313.63            | 308.01 | 315.88  |
| L32 B1024 | 85.97                 | 180.11                     | 176.51  | 176.91  | 51.06            | 51.63   | 51.13   | 99.10             | 97.21  | 99.50   |
| L32 B2048 | 165.14                | 313.90                     | 310.85  | 311.32  | 71.96            | 72.46   | 72.02   | 172.54            | 165.46 | 168.49  |
| L32 B4096 | 327.65                | 584.52                     | 580.88  | 581.78  | 114.01           | 114.57  | 113.91  | 320.63            | 311.32 | 324.41  |
| L40 B1024 | 89.35                 | 188.17                     | 184.87  | 185.69  | 58.24            | 58.92   | 58.55   | 106.71            | 100.61 | 108.18  |
| L40 B2048 | 171.95                | 326.02                     | 322.29  | 322.66  | 80.68            | 81.34   | 80.92   | 181.16            | 171.87 | 181.80  |
| L40 B4096 | 337.69                | 595.64                     | 591.08  | 592.07  | 124.07           | 124.56  | 124.23  | 331.25            | 313.98 | 335.31  |

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