
python convert_gpt2_checkpoint.py --model-name checkpoints/gpt2 --save-dir ./

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true \
--task-name wikitext \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 500000 --lr 1e-5 --seq-length 512 --batch-size 16 \
--forward-compress-method none \
--forward-scale-method max --forward-bits 2 --forward-bits-act 4 --forward-ratio-act 0.4 \
--backward-compress-method none --backward-ratio 0.2 \
--dist-url tcp://127.0.0.1:9032 \
--world-size 4 --pipeline-group-size 4 \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 2 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &