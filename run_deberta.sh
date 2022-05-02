
# python convert_deberta_checkpoint.py --model-name checkpoints/deberta-v3-base --save-dir ./

ARGS="--model-name ./checkpoints/deberta-v3-base \
--tokenizer-name ./checkpoints/deberta-v3-base-tokenizer \
--load-pretrained-model true \
--task-name cola \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 500000 --lr 5e-5 --seq-length 128 --batch-size 32 \
--forward-compress-method none \
--forward-scale-method max --forward-bits 8 --forward-bits-act 4 --forward-ratio 0.05 \
--backward-compress-method none --forward-bits 4 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9034 \
--world-size 4 --pipeline-group-size 4 \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &