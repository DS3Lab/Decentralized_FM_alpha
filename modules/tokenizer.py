
from transformers import GPT2TokenizerFast


def build_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
    