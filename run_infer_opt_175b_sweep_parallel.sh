PID=1049458
while s=`ps -p $PID -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
    sleep 1
done


# file=./wiki_test/wikitext_test_bloom_topk.jsonl


# export PARALLEL_BLOCKS=0

# output_file=./wiki_test/output_wikitext_test_opt175_pb${PARALLEL_BLOCKS}_att_mlp_10-86.jsonl
# eval_file=./wiki_test/eval_wikitext_test_opt175_pb${PARALLEL_BLOCKS}_att_mlp_10-86.jsonl

# echo "start running ${file}"

# ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
# --model-type opt-baseline \
# --seed 42 \
# --fp16 \
# --num-layers 12 \
# --max-layers 96 \
# --budget 18800 \
# --num-iters 1000 \
# --dist-url tcp://127.0.0.1:9131 \
# --token-micro-batch-size 2 \
# --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

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

# python -c "import json
# import numpy as np

# logprobs = []

# with open('$output_file') as f:
#     for line in f:
#         if line.strip() == '':
#             continue
#         if 'result' not in json.loads(line):
#             break
#         item = json.loads(line)

#         logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
# mean_logprob = sum(logprobs) / len(logprobs)
# perplexity = np.exp(-mean_logprob)
# print('perplexity:', perplexity)" > $eval_file
# cat $eval_file


# file=./c4_val/c4_val_opt_175b.jsonl


# export PARALLEL_SUBBLOCKS=0
# export PARALLEL_BLOCKS=0

# output_file=./c4_val/output_c4_val_opt175.jsonl
# eval_file=./c4_val/eval_c4_val_opt175.txt

# echo "start running ${file}"

# ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
# --model-type opt-baseline \
# --seed 42 \
# --fp16 \
# --num-layers 12 \
# --max-layers 96 \
# --budget 18800 \
# --num-iters 200 \
# --dist-url tcp://127.0.0.1:9131 \
# --token-micro-batch-size 2 \
# --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

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

# python -c "import json
# import numpy as np

# logprobs = []

# with open('$output_file') as f:
#     for line in f:
#         if line.strip() == '':
#             continue
#         if 'result' not in json.loads(line):
#             break
#         item = json.loads(line)

#         logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
# mean_logprob = sum(logprobs) / len(logprobs)
# perplexity = np.exp(-mean_logprob)
# print('perplexity:', perplexity)" > $eval_file
# cat $eval_file





# file=./c4_val/c4_val_opt_175b.jsonl


# export PARALLEL_SUBBLOCKS=1
# export PARALLEL_BLOCKS=0

# output_file=./c4_val/output_c4_val_opt175_pb${PARALLEL_BLOCKS}_att_mlp_10-86.jsonl
# eval_file=./c4_val/eval_c4_val_opt175_pb${PARALLEL_BLOCKS}_att_mlp_10-86.txt

# echo "start running ${file}"

# ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
# --model-type opt-baseline \
# --seed 42 \
# --fp16 \
# --num-layers 12 \
# --max-layers 96 \
# --budget 18800 \
# --num-iters 200 \
# --dist-url tcp://127.0.0.1:9132 \
# --token-micro-batch-size 2 \
# --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

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

# python -c "import json
# import numpy as np

# logprobs = []

# with open('$output_file') as f:
#     for line in f:
#         if line.strip() == '':
#             continue
#         if 'result' not in json.loads(line):
#             break
#         item = json.loads(line)

#         logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
# mean_logprob = sum(logprobs) / len(logprobs)
# perplexity = np.exp(-mean_logprob)
# print('perplexity:', perplexity)" > $eval_file
# cat $eval_file


# top


file=./wiki_test/wikitext_test_bloom_topk.jsonl


export PARALLEL_BLOCKS=2

output_file=./wiki_test/output_wikitext_test_opt175_pb${PARALLEL_BLOCKS}_10-86.jsonl
eval_file=./wiki_test/eval_wikitext_test_opt175_pb${PARALLEL_BLOCKS}_.txt

echo "start running ${file}"

ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
--model-type opt-baseline \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 96 \
--budget 18800 \
--num-iters 100 \
--dist-url tcp://127.0.0.1:9131 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

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

python -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('perplexity:', perplexity)" > $eval_file
cat $eval_file




export PARALLEL_BLOCKS=3

output_file=./wiki_test/output_wikitext_test_bloom_pb${PARALLEL_BLOCKS}_25-45.jsonl
eval_file=./wiki_test/eval_wikitext_test_bloom_pb${PARALLEL_BLOCKS}_25-45.txt

echo "start running ${file}"

ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
--model-type opt-baseline \
--seed 42 \
--fp16 \
--num-layers 12 \
--max-layers 96 \
--budget 18800 \
--num-iters 100 \
--dist-url tcp://127.0.0.1:9231 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

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

python -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print('perplexity:', perplexity)" > $eval_file
cat $eval_file


top





# export PARALLEL_BLOCKS=4

# output_file=./wiki_test/output_wikitext_test_bloom_pb${PARALLEL_BLOCKS}_25-45.jsonl
# eval_file=./wiki_test/eval_wikitext_test_bloom_pb${PARALLEL_BLOCKS}_25-45.txt

# echo "start running ${file}"

# ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
# --model-type opt-baseline \
# --seed 42 \
# --fp16 \
# --num-layers 12 \
# --max-layers 96 \
# --budget 18800 \
# --num-iters 100 \
# --dist-url tcp://127.0.0.1:9331 \
# --token-micro-batch-size 2 \
# --world-size 8 --pipeline-group-size 8 --data-group-size 1 \
# --pp-mode pipe_sync_sample_mask_token_pipe \
# --infer-data ${file} \
# --output-path ${output_file}"

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

# python -c "import json
# import numpy as np

# logprobs = []

# with open('$output_file') as f:
#     for line in f:
#         if line.strip() == '':
#             continue
#         if 'result' not in json.loads(line):
#             break
#         item = json.loads(line)

#         logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
# mean_logprob = sum(logprobs) / len(logprobs)
# perplexity = np.exp(-mean_logprob)
# print('perplexity:', perplexity)" > $eval_file
# cat $eval_file



sult']['choices'][0]['logprobs']['token_logprobs'][1:]
# mean_logprob = sum(logprobs) / len(logprobs)
# perplexity = np.exp(-mean_logprob)
# print('perplexity:', perplexity)" > $eval_file
# cat $eval_file



