
# python convert_hf_checkpoint.py --model-name gpt2-medium --save-dir ./checkpoints

ARGS="--model-name ./checkpoints/gpt2 --load-pretrained-model true \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 500 --lr 1e-4 --seq-length 128 \
--forward-compress-method delta-lowbits \
--forward-scale-method max --forward-bits 2 --forward-bits-act 4 \
--backward-compress-method topk --backward-ratio 0.2 \
--dist-url tcp://127.0.0.1:9030 \
--world-size 4 --pipeline-group-size 4 \
--pp-mode gpipe"

(trap 'kill 0' SIGINT; \
python dist_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 0 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 1 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &