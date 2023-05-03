
job_id=0

netif=lo
ip="127.0.0.1"

export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
export SHOW_DATA=1
export WANDB_NAME=pythia-codegen

main_program=dist_lm_pretrain.py

ARGS="--model-name /root/fm/models/pythia-t-chat-v0.16 \
--tokenizer-name /root/fm/models/pythia-t-chat-v0.16 \
--project-name codegen \
--optimizer adam \
--model-type gptneox \
--seed 42 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name \
/root/ft_data/codegen.jsonl:1.0 \
--num-layers 8 --embedding-dim 4096 \
--total-steps 15000 --warmup-steps 200 --train-warmup-steps 0 \
--checkpoint-steps 1000 \
--lr 1e-6 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 4 \
--dist-url tcp://${ip}:9117 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
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
