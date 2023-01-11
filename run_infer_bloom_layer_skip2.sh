# file=../lm-eval-harness-adapter/winogrande.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_winogrande_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)



# file=../lm-eval-harness-adapter/openbookqa.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_openbookqa_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/rte.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_rte_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/piqa.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_piqa_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/copa.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_copa_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/hellaswag.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_hellaswag_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/lambada_openai.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_lambada_openai_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/wic.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_wic_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

# file=../lm-eval-harness-adapter/wsc.jsonl
# export SKIP_BLOCKS=2
# output_file=../lm-eval-harness-adapter/output_bloom_wsc_skip${SKIP_BLOCKS}.jsonl
# echo "start running ${file}"
# ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
# --model-type bloom \
# --seed 42 \
# --fp16 \
# --num-layers 10 \
# --max-layers 70 \
# --budget 10800 \
# --num-iters 100000000 \
# --dist-url tcp://127.0.0.1:9031 \
# --token-micro-batch-size 2 \
# --world-size 7 --pipeline-group-size 7 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

# (trap 'kill 0' SIGINT; \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
#     & \
# python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
#     & \
# wait)

file=../lm-eval-harness-adapter/cb.jsonl
export SKIP_BLOCKS=2
output_file=../lm-eval-harness-adapter/output_bloom_cb_skip${SKIP_BLOCKS}.jsonl
echo "start running ${file}"
ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
--model-type bloom \
--seed 42 \
--fp16 \
--num-layers 10 \
--max-layers 70 \
--budget 10800 \
--num-iters 100000000 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 2 \
--world-size 7 --pipeline-group-size 7 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 0 \
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
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
wait)