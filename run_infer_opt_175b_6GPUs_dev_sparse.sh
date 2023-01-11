export THRESHOLD=0.4

ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
--model-type opt-classifier-sparse \
--seed 42 \
--fp16 \
--num-layers 16 \
--max-layers 96 \
--num-iters 1 \
--dist-url tcp://127.0.0.1:9031 \
--batch-size 32 \
--input-seq-length 128 \
--generate-seq-length 8 \
--token-micro-batch-size 1 \
--world-size 6 --pipeline-group-size 6 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
wait)

