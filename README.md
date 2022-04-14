# GPT-home-private

## Setup:

- Use AWS Deep Learning Base AMI


- Install PyTorch env: 

      pip3 install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html

      # Magic, not sure why cupy-cuda111 would not work, it seems that cupy-cuda111 will use different PTX from torch.
      pip3 install cupy-cuda110==8.6.0

- Clone this repo:
        
      git clone https://github.com/BinhangYuan/GPT-home-private.git

- set the github cache (Optional):

      git config credential.helper 'cache --timeout=30000'

- Download a tiny dataset:

      wget https://binhang-language-datasets.s3.us-west-2.amazonaws.com/glue_qqp_dataset/data.tar.xz -P ./glue_dataset/
      
      tar -xvf ./glue_dataset/data.tar.xz -C ./glue_dataset/

- Setup network configuration:

      export GLOO_SOCKET_IFNAME=ens3

      export NCCL_SOCKET_IFNAME=ens3


- Use TC scripts to control network delay and bandwidth:
  

## Run Distributed Gpipe:

- On each node, run:
      
      python dist_pipeline_runner.py --dist-url tcp://XXX.XXX.XXX.XXX:9000 --world-size N --rank i (i=0,...,N-1)

## Run with Advanced Scripts (under scripts directory):

- First update the public IPs and private IP of the rank-0 node in ip_list.sh.

- Allow SSH connects: 

      bash accept_ssh_keys.sh

- Update local repository:

      bash aws_sync_code.sh #GIT_TOKEN
      
- Enable environment: (This is optional but load conda env seems to be slow for the first time)

      bash aws_foo_load_lib.sh

- Run Tasks (e.g.,):

      bash aws_run_gpt3_training.sh gpt3_small_pp3_dp4.sh
      bash aws_run_gpt3_training.sh gpt3_small_pp3_dp4.sh #DELAY #BANDWIDTH

- Clear logs:

      bash aws_clear_logs.sh

- Copy training logs from Rank-0 node (For my implementation the benchmark result is on the rank-0 node.)

      bash copy_rank0_logs.sh

- Download and generate trace:

      bash copy_traces.sh #PREFIX
      bash generate_traces.sh #PREFIX

