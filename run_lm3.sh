
for lr in 5
do

for drop_iters in 10
do

for p in 0.02
do

ARGS="--model-name ./checkpoints/gpt2 \
--tokenizer-name ./checkpoints/gpt2-tokenizer \
--load-pretrained-model true --warmup-epochs 1 --n-epochs 10 \
--task-name wikitext --seed 4242 \
--layer-drop-p ${p} \
--layer-drop-iters ${drop_iters} \
--layer-drop-method sample \
--num-layers 3 --num-heads 12 --embedding-dim 768 \
--num-iters 100000000 --lr ${lr}e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--dist-url tcp://127.0.0.1:9035 \
--world-size 4 --pipeline-group-size 4 --data-group-size 1 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    &
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
wait)

# > /dev/null 2>&1 &

done
done
done
