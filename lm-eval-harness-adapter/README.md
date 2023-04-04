# LM-eval

This is an adapter of `lm-evaluation-harness`.

# QuickStart

1. Generate Input Data

```bash
python generate_task_data.py --output-file wsc.jsonl --task-name wsc --num-fewshot 0
```
Here we use WSC task.

2. Do inference

A sample code to do inference, change the `input_path`， `output_path`， and the model.

`input_path` is the path of the data generated in Step 1.
`output_path` is the path of the inference result.

```python
import json, tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

input_path = 'wsc.jsonl'
output_path = 'wsc_out.jsonl'

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").half().eval().to('cuda:0')


requests = []
with open(input_path, 'r') as f:
    for line in f:
        if line.strip() != '':
            requests.append(json.loads(line))

results = []
with torch.no_grad():
    for request in tqdm.tqdm(requests):
        result = {'request': request, 'result': {}}
        prompt = request['prompt']
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

        logits = model(input_ids).logits.log_softmax(dim=-1)

        values, indices = logits.squeeze(0).topk(dim=-1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        gold_indices = input_ids[:, 1:] # skip first
        logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
        top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
        
        result['result'] = {
            "choices": [
                {
                    "text": prompt, 
                    "logprobs": {
                        "tokens": tokens, 
                        "token_logprobs": logprobs, 
                        "top_logprobs": top_logprobs, 
                        "text_offset": []
                    }, 
                    "finish_reason": "length"
                }
            ], 
            "request_time": {
                "batch_time": 0, 
                "batch_size": 1}
        }
        
        results.append(result)

with open(output_path, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
```

3. Eval Outputs

```bash
python evaluate_task_result.py --result-file wsc_out.jsonl --task-name wsc --num-fewshot 0 --model-type opt
```
