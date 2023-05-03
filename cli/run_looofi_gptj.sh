job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

netif=access
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}

main_program=dist_lm_pretrain.py

ARGS="--model-name /root/fm/models/gpt-j-6B \
--tokenizer-name /root/fm/models/gpt-j-6B \
--model-type gptj \
--seed 42 \
--checkpoint-path ./model_checkpoints/gptj-test2 \
--load-pretrained-model false \
--task-name pile \
--num-layers 1 --num-heads 16 --embedding-dim 4096 \
--total-steps 100000 --warmup-steps 100 --train-warmup-steps 0 \
--checkpoint-steps 10 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9011 \
--world-size 4 --pipeline-group-size 2 --data-group-size 2 \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python ${main_program} $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 7 --rank 1 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 7 --rank 2 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 6 --rank 3 \
    & \
wait)
