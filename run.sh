NNode=3
MODE=gpipe
# MODE=1f1b

python convert_hf_checkpoint.py --model-name gpt2 --save-dir ./checkpoints

(trap 'kill 0' SIGINT; \
python dist_runner.py \
    --model-name ./checkpoints/gpt2 \
    --cuda-id 5 --dist-url tcp://127.0.0.1:9030 \
    --world-size $NNode --pipeline-group-size $NNode --rank 0 \
    --pp-mode ${MODE} \
    & \
python dist_runner.py \
    --model-name ./checkpoints/gpt2 \
    --cuda-id 6 --dist-url tcp://127.0.0.1:9030 \
    --world-size $NNode --pipeline-group-size $NNode --rank 1 \
    --pp-mode ${MODE} \
    & \
python dist_runner.py \
    --model-name ./checkpoints/gpt2 \
    --cuda-id 7 --dist-url tcp://127.0.0.1:9030 \
    --world-size $NNode --pipeline-group-size $NNode --rank 2 \
    --pp-mode ${MODE} \
    & \
wait)


# > /dev/null 2>&1 &