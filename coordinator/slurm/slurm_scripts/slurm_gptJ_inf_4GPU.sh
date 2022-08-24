#!/bin/bash
#SBATCH --job-name=gptJ_inf_4GPU
#
#SBATCH --gres gpu:titanx:1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/exe_log/gptJ_inf_4GPU_%j.log
port=$1

source activate base                          # Activate my conda python environment
cd /afs/cs.stanford.edu/u/biyuan/GPT-home-private     # Change directory

nvidia-smi

ifconfig

world_size=4

DIST_CONF="--pp-mode pipe_sync_greedy --pipeline-group-size $world_size --cuda-id 0"
MODEL_CONF="--model-type gptj --model-name ./pretrained_models/gpt-j-6B --num-iters 10"
INFERENCE_CONF="--fp16 --batch-size 24 --input-seq-length 512 --generate-seq-length 32 --micro-batch-size 1 --num-layers 7"
COOR_CONF="--coordinator-server-ip 10.79.12.70  --unique-port $port"

#export NCCL_SOCKET_IFNAME=ens160
#export GLOO_SOCKET_IFNAME=ens160

python dist_inference_runner_w_slurm_coordinator.py $DIST_CONF $MODEL_CONF $INFERENCE_CONF $COOR_CONF