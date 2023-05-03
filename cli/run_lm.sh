
# python convert_gpt2_checkpoint.py --model-name checkpoints/gpt2 --save-dir ./

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model false --n-epochs 10 \
--task-name arxiv21 \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method none \
--forward-scale-dims (1,) \
--forward-scale-method max \
--forward-bits 3 --forward-bits-act 4 --forward-ratio 0.05 \
--backward-compress-method none \
--backward-scale-dims (1,) \
--backward-bits 6 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 4 --pipeline-group-size 4 --data-group-size 1 \
--dp-mode sharded_ps_compressed --dp-bits 4 \
--pp-mode gpipe --profiling no-profiling --do-evaluation false"

(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 4 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 5 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 6 \
#     & \
# python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 7 \
#     & \
wait)

# > /dev/null 2>&1 &
