# file=./c4_val/c4_val_bloom_sparse.jsonl
# output_file=./c4_val/output_c4_val_bloom_sparse.jsonl
# eval_file=./c4_val/eval_c4_val_bloom_sparse.txt
file=./wiki_test/wikitext_test_bloom_topk.jsonl
output_file=./wiki_test/output_wikitext_test_bloom_topk.jsonl
eval_file=./wiki_test/eval_wikitext_test_bloom_relu.txt
    
echo "start running ${file}"

ARGS="--model-name /mnt/workspace/checkpoint/bloom-new \
--model-type bloom-relu \
--seed 42 \
--fp16 \
--num-layers 9 \
--max-layers 70 \
--budget 22000 \
--num-iters 100 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 2 \
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
