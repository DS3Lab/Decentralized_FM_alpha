# Run ZeRO Comparison

### Install deepspeed

- Install some toolkit:

      sudo apt-get install pdsh
      sudo apt-get install libaio-dev

- Install conda:
       
      wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
      
      bash Anaconda-latest-Linux-x86_64.sh
      
      (re-connect to the node to activate conda)
    
- Install PyTorch:

       conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

- Install Deepspeed:

       pip3 install deepspeed

### Config deepspeed.
- Setup ssh access:

      (copy the .pem file to the client)       
      
      eval `ssh-agent`
 
      ssh-add  binhang_ds3_aws_oregon.pem


- Config the hostfile:

  - Put IP as the hostname for each node. 
  - Set slots=1 for P3.2xlarge (1 GPU per node).


- Config the ZeRO related parameters

  - Change the zero_dp_s3_config.json file


### Run ZeRO:

- Follow the example in run_zero.sh


### Result