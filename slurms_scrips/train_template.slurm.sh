#!/bin/bash

#SBATCH --job-name=gpt_j_6b
#SBATCH --gpus=gtx_1080:1
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/exec_log/gpt_j_6b_slurm_%j.log

# `python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=$1 
pp_degree=$2
dp_degree=$3
n_layer_per_device=$4

world_size=`expr $pp_degree \* $dp_degree`

netif=access
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

root_path=/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm

main_program=dist_lm_pretrain.py

ARGS="--model-name ${root_path}/pretrained_models/gpt-j-6B \
--tokenizer-name ${root_path}/pretrained_models/gpt-j-6B \
--model-type gptj \
--seed 42 \
--checkpoint-path ${root_path}/pretrained_models/checkpoints/gptj-test \
--load-pretrained-model false \
--task-name pile \
--num-layers ${n_layer_per_device} --num-heads 16 --embedding-dim 4096 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 10 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size ${world_size} --pipeline-group-size ${pp_degree} --data-group-size ${dp_degree} \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

python ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 # rank will be rewriten
