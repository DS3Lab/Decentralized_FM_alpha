# GPT-home-private

## Setup:

- Use AWS Deep Learning Base AMI


- Install PyTorch env: 

      pip3 install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html

      # Magic, not sure why cupy-cuda111 would not work, it seems that cupy-cuda111 will use different PTX from torch.
      pip3 install cupy-cuda110==8.6.0

- Clone this repo:
        
      https://github.com/BinhangYuan/GPT-home-private.git

- set the github cache (Optional) 

      git config credential.helper 'cache --timeout=30000'

## Run Distributed Gpipe:

- On each node, run:
      
      python3 dist_gpipe_runner.py --dist-url tcp://XXX.XXX.XXX.XXX:9000 --world-size N --rank i (i=0,...,N-1)
