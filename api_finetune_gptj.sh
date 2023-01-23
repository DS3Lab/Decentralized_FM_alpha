project_id=$PROJECT_ID
dataset_url=$DATASET_URL

echo $project_id

export NCCL_DEBUG=INFO
export WANDB_DISABLED=1
huggingface-cli repo create ${project_id} -y
git clone https://huggingface.co/xzyao/${project_id} ./model_checkpoints/${project_id}

wget ${dataset_url} -O data/${project_id}.jsonl

main_program=dist_lm_pretrain.py

total_steps=$TOTAL_STEPS

ARGS="--model-name /mnt/ds3lab-scratch/fm/pretrained_models/gpt-j-6B \
--tokenizer-name /mnt/cc/fm/pretrained_models/gpt-j-6B \
--project-name ${project_id} \
--model-type gptj \
--seed 42 \
--checkpoint-path ./model_checkpoints/${project_id} \
--load-pretrained-model true \
--task-name data/${project_id}.jsonl \
--num-layers 1 --num-heads 32 --embedding-dim 4096 \
--total-steps ${total_steps} --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 1e-5 --seq-length 2048 --batch-size 5 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--job-id ${project_id} --net-interface enp13s0f0 \
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

FINETUNE_ID=${project_id} python tohub.py