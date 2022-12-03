
import os
import uuid

template = '''#!/bin/bash
#BSUB -n 4                     # 1 cores
#BSUB -W 3:59                   # 3-minutes run-time
#BSUB -R "rusage[mem=8000]"     # 32 GB per core
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=20000]"
#BSUB -o /cluster/home/juewang/fm/juewang/exe_log/out.%J
#BSUB -e /cluster/home/juewang/fm/juewang/exe_log/err.%J

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
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
export WANDB_DISABLE_SERVICE=1

root_path=/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm

main_program=dist_lm_pretrain.py

ARGS="--model-name ${root_path}/pretrained_models/opt-1.3b-new \
--tokenizer-name ${root_path}/pretrained_models/opt-1.3b-new \
--project-name loooofi \
--model-type opt \
--seed 42 \
--checkpoint-path ${root_path}/pretrained_models/checkpoints/opt-looofi \
--load-pretrained-model true \
--task-name /cluster/home/juewang/scratch/pile_1280k.jsonl:0.9,ni:0.1 \
--num-layers ${n_layer_per_device} --num-heads 32 --embedding-dim 2048 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 2e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size ${world_size} --pipeline-group-size ${pp_degree} --data-group-size ${dp_degree} \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-mode local \
--pp-mode gpipe --profiling no-profiling"

python -u ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 # rank will be rewriten
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

    with open('slurms_scrips/train_to_submmit.lsf.sh', 'w') as f:
        f.write(template)
        
    for i in range(world_size):
        os.system('bsub < slurms_scrips/train_to_submmit.lsf.sh')
    