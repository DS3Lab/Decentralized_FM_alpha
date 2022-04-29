
# python convert_hf_checkpoint.py --model-name gpt2-medium --save-dir ./checkpoints

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true \
--task-name cola \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 5000 --lr 1e-4 --seq-length 128 \
--forward-compress-method delta-topk-lowbits \
--forward-scale-method max --forward-bits 2 --forward-bits-act 4 --forward-ratio-act 0.1 \
--backward-compress-method topk --backward-ratio 0.2 \
--dist-url tcp://127.0.0.1:9031 \
--world-size 4 --pipeline-group-size 4 \
--pp-mode gpipe"

(trap 'kill 0' SIGINT; \
python dist_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &