
job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
job_id=0

ARGS="--model-name /root/fm/models/opt-1.3b-new \
--tokenizer-name /root/fm/models/opt-1.3b-new \
--model-type opt \
--seed 4242 \
--checkpoint-path ./model_checkpoints/opt_1.3b_ni_default \
--load-pretrained-model true \
--task-name natural_instructions \
--num-layers 12 --num-heads 32 --embedding-dim 2048 \
--total-steps 200000 --warmup-steps 100 \
--checkpoint-steps 100 \
--lr 1e-5 --seq-length 2048 --batch-size 64 --micro-batch-size 8 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id ${job_id} --net-interface lo \
--fp16 --loss-scale-window 100 \
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
