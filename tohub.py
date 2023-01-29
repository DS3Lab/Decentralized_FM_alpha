import os
from transformers import OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM

print("Uploading model to hub")
finetune_id = os.environ.get("FINETUNE_ID")
finetune_path = os.path.join("model_checkpoints", finetune_id)

model = AutoModelForCausalLM.from_pretrained(finetune_path)

tokenizer = AutoTokenizer.from_pretrained(finetune_path)

tokenizer.push_to_hub(
    repo_path_or_name=f"./model_checkpoints/{finetune_id}",
    repo_url=f"https://huggingface.co/xzyao/{finetune_id}",
    use_auth_token=True,
)
model.push_to_hub(
    repo_path_or_name=f"./model_checkpoints/{finetune_id}",
    repo_url=f"https://huggingface.co/xzyao/{finetune_id}",
    use_auth_token=True,
)
