
# python convert_hf_checkpoint.py --model-name checkpoints/gpt2-xl --save-dir ./

ARGS="--model-name ./checkpoints/gpt2-xl \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true \
--task-name cola \
--num-layers 6 --num-heads 25 --embedding-dim 1600 \
--num-iters 50000 --lr 1e-5 --seq-length 128 \
--forward-compress-method delta \
--forward-scale-method max --forward-bits 2 --forward-bits-act 4 --forward-ratio-act 0.4 \
--backward-compress-method topk --backward-ratio 0.2 \
--dist-url tcp://127.0.0.1:9031 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

# > /dev/null 2>&1 &