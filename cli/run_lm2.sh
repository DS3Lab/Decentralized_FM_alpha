
# python convert_gpt2_checkpoint.py --model-name checkpoints/gpt2 --save-dir ./

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true --warmup-epochs 1 --n-epochs 10 \
--task-name wikitext --seed 42 \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 100000000 --lr 1e-4 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method none \
--forward-scale-dims (0,1) \
--forward-scale-method max \
--forward-ratio 1 \
--forward-bits 4 \
--backward-compress-method none \
--backward-scale-dims (0,1) \
--backward-bits 8 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 4 --pipeline-group-size 4 --data-group-size 1 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 1 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 2 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &

