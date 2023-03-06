# generate uuid and ask rank server
job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
# project_id=`python3 -c 'import random;import string; print("".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(50)))'`
# job_id=0 will not ask rank server but use the rank given in the argument
# job_id=0
project_id=$PROJECT_ID
dataset_url=$DATASET_URL
learning_rate=$LEARNING_RATE

echo $project_id

export NCCL_DEBUG=INFO
export WANDB_DISABLED=1
huggingface-cli repo create ${project_id} -y
git clone https://huggingface.co/xzyao/${project_id} ./model_checkpoints/${project_id}

wget ${dataset_url} -O data/${project_id}.jsonl

main_program=dist_lm_pretrain.py

total_steps=$TOTAL_STEPS

ARGS="--model-name /mnt/ds3lab-scratch/fm/pretrained_models/opt-1.3b-new \
--tokenizer-name /mnt/ds3lab-scratch/fm/pretrained_models/opt-1.3b-new \
--project-name ${project_id} \
--model-type opt \
--seed 42 \
--checkpoint-path ./model_checkpoints/${project_id} \
--load-pretrained-model true \
--task-name data/${project_id}.jsonl \
--num-layers 2 --num-heads 32 --embedding-dim 2048 \
--total-steps ${total_steps} --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps ${total_steps} \
--lr ${learning_rate} --seq-length 2048 --batch-size 5 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--job-id ${job_id} --net-interface enp13s0f0 \
--fp16 \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

echo $ARGS

(trap 'kill 0' SIGINT; \
python ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
wait)

cp /mnt/ds3lab-scratch/fm/pretrained_models/opt-1.3b-new/*.json ./model_checkpoints/${project_id}/
cp /mnt/ds3lab-scratch/fm/pretrained_models/opt-1.3b-new/*.txt ./model_checkpoints/${project_id}/
cp ./model_checkpoints/${project_id}/checkpoint_${total_steps}/* ./model_checkpoints/${project_id}/

FINETUNE_ID=${project_id} python tohub.py
