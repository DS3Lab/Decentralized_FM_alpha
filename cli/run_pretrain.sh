
job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`

ARGS="--model-name ./model_checkpoints/gpt2-config \
--tokenizer-name ./model_checkpoints/gpt2-tokenizer \
--checkpoint-path ./model_checkpoints/gpt2-base \
--load-pretrained-model false \
--task-name pile \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--total-steps 10000000 --warmup-steps 5000 \
--checkpoint-steps 5000 \
--lr 1e-3 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 4 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
--job-id ${job_id} --net-interface lo \
--fp16 \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_lm_pretrain.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

# > /dev/null 2>&1 &
