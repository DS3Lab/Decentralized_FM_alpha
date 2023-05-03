# while ps -p 54007; do sleep 1; done ; echo "starting"

# python convert_deberta_checkpoint.py --model-name checkpoints/deberta-v3-base --save-dir ./

ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name ./checkpoints/deberta-v2-xxl-tokenizer \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 20 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 1000000000000 --lr 2.5e-6 --seq-length 50 --batch-size 64 --micro-batch-size 8 \
--forward-compress-method none \
--forward-scale-method max \
--forward-bits 2 --forward-bits-act 8 \
--backward-compress-method none \
--backward-bits 4 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)


ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name ./checkpoints/deberta-v2-xxl-tokenizer \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 20 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 1000000000000 --lr 2.5e-6 --seq-length 50 --batch-size 64 --micro-batch-size 8 \
--forward-compress-method fixpoint \
--forward-scale-method max \
--forward-bits 2 --forward-bits-act 8 \
--backward-compress-method fixpoint \
--backward-bits 4 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)


ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name ./checkpoints/deberta-v2-xxl-tokenizer \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 20 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 1000000000000 --lr 2.5e-6 --seq-length 50 --batch-size 64 --micro-batch-size 8 \
--forward-compress-method fixpoint \
--forward-scale-method max \
--forward-bits 3 --forward-bits-act 8 \
--backward-compress-method fixpoint \
--backward-bits 6 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)



ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name ./checkpoints/deberta-v2-xxl-tokenizer \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 20 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 1000000000000 --lr 2.5e-6 --seq-length 50 --batch-size 64 --micro-batch-size 8 \
--forward-compress-method delta \
--forward-scale-method max \
--forward-bits 2 --forward-bits-act 8 \
--backward-compress-method fixpoint \
--backward-bits 4 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)


ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name ./checkpoints/deberta-v2-xxl-tokenizer \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 20 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 1000000000000 --lr 2.5e-6 --seq-length 50 --batch-size 64 --micro-batch-size 8 \
--forward-compress-method delta \
--forward-scale-method max \
--forward-bits 3 --forward-bits-act 8 \
--backward-compress-method fixpoint \
--backward-bits 6 --backward-ratio 0.4 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)
