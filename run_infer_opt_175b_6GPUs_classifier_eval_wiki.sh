file=./wiki_test/wikitext_test_opt_175b_classifier.jsonl
output_file=./wiki_test/output_wikitext_test_opt_175b_classifier.jsonl
eval_file=./wiki_test/eval_wikitext_test_opt_175b_classifier.jsonl

export THRESHOLD=0.4
    
echo "start running ${file}"

ARGS="--model-name /mnt/workspace/checkpoint/opt-175b-new \
--model-type opt-classifier-sparse \
--seed 42 \
--fp16 \
--num-layers 16 \
--max-layers 96 \
--budget 10800 \
--num-iters 1000 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 2 \
--world-size 6 --pipeline-group-size 6 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 1 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 2 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 3 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 4 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 5 \
    & \
wait)

python -c "import json
import numpy as np

logprobs = []

with open('$output_file') as f:
    n = 0
    for line in f:
        if line.strip() == '':
            continue
        if 'result' not in json.loads(line):
            break
        item = json.loads(line)

        logprobs += item['result']['choices'][0]['logprobs']['token_logprobs'][1:]
        n += 1
mean_logprob = sum(logprobs) / len(logprobs)
perplexity = np.exp(-mean_logprob)
print(f'perplexity:{ perplexity}, number of data:{n}')" > $eval_file
