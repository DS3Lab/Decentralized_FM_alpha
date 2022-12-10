
import os
import uuid

template = '''#!/bin/bash
#SBATCH --job-name=gpt_j_6b
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/home/juewang/fm/juewang/exe_log/gpt_j_6b_slurm_%j.log

module load gcc/6.3.0 cuda/11.0.3 eth_proxy       # Load modules from Euler setup
source activate pipeline                          # Activate my conda python environment
cd /cluster/home/juewang/fm/juewang/Decentralized_FM_alpha_train     # Change directory

nvidia-smi

# `python3 -c 'import uuid; print(uuid.uuid4())'`
job_id={{JOB_ID}}
pp_degree={{PP_DEGREE}}
dp_degree={{DP_DEGREE}}
n_layer_per_device={{N_LAYER_PER_DEVICE}}

world_size=`expr $pp_degree \* $dp_degree`

netif=access
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export WANDB_DISABLE_SERVICE=1

root_path=/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm

main_program=dist_lm_pretrain.py

ARGS="--model-name ${root_path}/pretrained_models/opt-1.3b-new \
--tokenizer-name ${root_path}/pretrained_models/opt-1.3b-new \
--project-name loooofi \
--model-type opt \
--seed 4242 \
--checkpoint-path ${root_path}/pretrained_models/checkpoints/opt-afreeze-20x-new \
--load-pretrained-model true \
--task-name /cluster/home/juewang/scratch/pile_1280k.jsonl:0.5,ni:0.5 \
--num-layers ${n_layer_per_device} --num-heads 32 --embedding-dim 2048 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 1e-4 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size ${world_size} --pipeline-group-size ${pp_degree} --data-group-size ${dp_degree} \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-mode afreeze \
--pp-mode gpipe --profiling no-profiling"

python -u ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 # aprox
'''

if __name__ == '__main__':

    # with open('slurms_scrips/train_template.lsf.sh') as f:
    #     template = f.read()

    job_id = str(uuid.uuid4())
    pp_degree=3
    dp_degree=4
    n_layer_per_device=12
    world_size = pp_degree * dp_degree

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', str(pp_degree))
    template = template.replace('{{DP_DEGREE}}', str(dp_degree))
    template = template.replace('{{N_LAYER_PER_DEVICE}}', str(n_layer_per_device))

    with open('slurms_scrips/train_to_submit.slurm.sh', 'w') as f:
        f.write(template)
        
    for i in range(world_size):
        os.system('sbatch slurms_scrips/train_to_submit.slurm.sh')
    