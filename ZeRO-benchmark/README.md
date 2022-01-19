## Run ZeRO Comparison

### Install deepspeed

    pip3 install deepspeed

### Config the hostfile

- Put IP as the hostname for each node. 
- Set slots=1 for P3.2xlarge (1 GPU per node).

### Config the ZeRO related parameters

- Change the zero_dp_s3_config.json file


### Run ZeRO:

- Follow the example in run_zero.sh


## Result