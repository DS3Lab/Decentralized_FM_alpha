# export THRESHOLD=0.45


# file=lm_eval/opt-sparse/split0.jsonl

# echo "start running ${file}"

# ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
# --model-type opt-classifier-sparse \
# --seed 42 \
# --fp16 \
# --num-layers 12 \
# --max-layers 96 \
# --budget 20800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 1 \
# --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     &
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
#     & \
# wait)


file=lm_eval/opt-original/split0.jsonl

echo "start running ${file}"

ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
--model-type opt-classifier-sparse-bylayer \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 96 \
--budget 800 \
--num-iters 100000000 \
--dist-url tcp://127.0.0.1:9030 \
--token-micro-batch-size 1 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file}"

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
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)



