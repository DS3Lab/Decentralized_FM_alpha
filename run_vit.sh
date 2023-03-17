# PID=36351
# while s=`ps -p $PID -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
#     sleep 1
# done

job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

export SYNC_STEPS=10
export QUANT_BITS=4
export TOPK_RATIO=0.2

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
export WANDB_NAME=vit-large-imagenet-1k-allreduce
export WANDB_ENTITY=pipeline-activation-compression
export WANDB_MODE=online

main_program=dist_vit_finetune.py

ARGS="--model-name google/vit-large-patch32-384 \
--tokenizer-name google/vit-large-patch32-384 \
--project-name slot-sgd \
--optimizer adam \
--model-type vit \
--seed 4242 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name imagenet-1k \
--num-layers 1 --num-heads 1 --embedding-dim 1 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 20000 \
--lr 1e-5 --seq-length 384 --batch-size 32 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://${ip}:9317 \
--world-size 4 --pipeline-group-size 1 --data-group-size 4 \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode nopipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python ${main_program} $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &
