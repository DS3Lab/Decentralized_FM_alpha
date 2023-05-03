job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

export WANDB_ENTITY=asdfffjj
export WANDB_NAME=gptjt-safety-fix

main_program=dist_prefixlm_pretrain.py

ARGS="--model-name /root/fm/models/GPT-JT-6B-v1 \
--tokenizer-name /root/fm/models/GPT-JT-6B-v1 \
--model-type gptj \
--project-name safety \
--checkpoint-path ./model_checkpoints/gptjt-satefy-fix \
--load-pretrained-model true \
--task-name /root/ft_data/safety_prosocial_plus_normal.jsonl \
--num-layers 14 --num-heads 16 --embedding-dim 4096 \
--total-steps 20000 --warmup-steps 100 --train-warmup-steps 200 \
--checkpoint-steps 100 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9022 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id ${job_id} --net-interface lo \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

# > /dev/null 2>&1 &

