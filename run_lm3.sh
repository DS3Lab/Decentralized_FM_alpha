
# --dp-mode local \

# ARGS="--model-name ./checkpoints/gpt2 \
# --tokenizer-name ./checkpoints/gpt2-tokenizer \
# --load-pretrained-model true --warmup-epochs 1 --n-epochs 10 \
# --task-name arxiv21 \
# --num-layers 6 --num-heads 12 --embedding-dim 768 \
# --num-iters 10000000 --lr 1e-4 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
# --forward-compress-method none \
# --backward-compress-method none \
# --dist-url tcp://127.0.0.1:9033 \
# --world-size 8 --pipeline-group-size 2 --data-group-size 4 \
# --pp-mode gpipe --profiling no-profiling --do-evaluation true"

# (trap 'kill 0' SIGINT; \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
#     & \
# wait)

# # > /dev/null 2>&1 &

PID=533359
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running"
    sleep .6
done

# --dp-mode local \

job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true --warmup-epochs 2 --n-epochs 10 \
--task-name arxiv21 \
--num-layers 6 --num-heads 12 --embedding-dim 768 \
--num-iters 10000000 --lr 1e-4 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method none \
--backward-compress-method none \
--dp-mode proxskip_adam \
--fp16 --loss-scale 0 --initial-loss-scale 4096 --loss-scale-window 100 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id ${job_id} --net-interface lo \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_lm_runner_w_coordinator.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

# > /dev/null 2>&1 &


