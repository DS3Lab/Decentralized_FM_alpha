ARGS="--model-name /mnt/workspace/checkpoint/opt-30b-new \
--model-type opt-flash \
--seed 42 \
--fp16 \
--num-layers 24 \
--max-layers 48 \
--num-iters 10 \
--dist-url tcp://127.0.0.1:9001 \
--batch-size 4 \
--input-seq-length 1024 \
--generate-seq-length 32 \
--token-micro-batch-size 2 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 1 \
    & \
wait)

