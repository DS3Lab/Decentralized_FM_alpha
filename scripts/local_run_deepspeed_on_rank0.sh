cd ~/GPT-home-private
source activate pytorch_p38
export NCCL_SOCKET_IFNAME=ens3

case=$1
layer=$2
batch=$3

echo "Running case ${case} layer ${layer} batch ${batch}"


timestamp=$(date +%Y_%m_%d_%H_%M)
deepspeed --hostfile=./scripts/ds_hostnames dist_deepspeed_zero_s3.py  --micro-batch-size 4 --embedding-dim 2048 --seq-length 2048 --batch-size $batch --num-layers $layer >> "./logs/${timestamp}_deepspeed_case${case}_L${layer}_B${batch}.log"