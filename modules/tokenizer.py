
from transformers import GPT2TokenizerFast


def build_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
    