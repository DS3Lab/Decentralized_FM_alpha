
from transformers import GPT2TokenizerFast


def build_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return tokenizer