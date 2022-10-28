job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

main_program=dist_prefix_pretrain.py

ARGS="--model-name /root/fm/models/gpt-j-6B \
--tokenizer-name /root/fm/models/gpt-j-6B \
--checkpoint-path ./model_checkpoints/gptj-baseline \
--load-pretrained-model true \
--task-name pile \
--num-layers 1 --num-heads 16 --embedding-dim 4096 \
--total-steps 20000 --warmup-steps 100 --train-warmup-steps 200 \
--checkpoint-steps 100 \
--lr 1e-6 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9022 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id ${job_id} --net-interface lo \
--fp16 --loss-scale-window 100 \
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

