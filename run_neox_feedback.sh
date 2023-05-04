
job_id=0

netif=lo
ip="127.0.0.1"

export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO

export SHOW_DATA=1
export WANDB_NAME=gpt-neoxt-ock-feedback

main_program=dist_lm_pretrain.py

ARGS="--model-name /root/fm/models/GPT-NeoXT-20B-chat-v0.16-3.8K \
--tokenizer-name /root/fm/models/GPT-NeoXT-20B-chat-v0.16-3.8K \
--project-name slot-sgd \
--optimizer adam \
--model-type gptneox \
--seed 42 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name /root/ft_data/unified_feedback_v0.2.jsonl \
--num-layers 6 --embedding-dim 6144 \
--total-steps 400 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 10 \
--lr 1e-6 --seq-length 2048 --batch-size 64 --micro-batch-size 1 --gradient-accumulate-step 2 \
--dist-url tcp://${ip}:9017 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--job-id ${job_id} --net-interface ${netif} \
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

