# PID=36351
# while s=`ps -p $PID -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
#     sleep 1
# done

job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

export SYNC_STEPS=100
export QUANT_BITS=4
export TOPK_RATIO=1

netif=lo
ip="127.0.0.1"
# netif=p2ptun
# ip="10.133.70.1"
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
# export NCCL_PROTO=SIMPLE
# export NCCL_DEBUG_SUBSYS=COLL
export SHOW_DATA=0
export WANDB_NAME=opt-wiki-powersgd-debug-new
export WANDB_ENTITY=pipeline-activation-compression
export WANDB_MODE=online
# export WANDB_MODE=offline

main_program=dist_lm_pretrain.py

ARGS="--model-name /root/fm/models/opt-1.3b-new \
--tokenizer-name /root/fm/models/opt-1.3b-new \
--project-name slot-sgd \
--optimizer adam \
--model-type opt \
--seed 4242 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name /root/ft_data/wiki103.jsonl \
--num-layers 12 --num-heads 32 --embedding-dim 2048 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 20000 \
--lr 1e-4 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://${ip}:9017 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode powersgd \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 0 --rank 4 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 1 --rank 5 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 2 --rank 6 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 3 --rank 7 \
    & \
wait)

# > /dev/null 2>&1 &
