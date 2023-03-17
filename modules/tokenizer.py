
from transformers import ViTFeatureExtractor # ViTImageProcessor
from transformers import AutoTokenizer, GPT2TokenizerFast, DebertaV2Tokenizer
from .llama_modules import LLaMATokenizer

def build_tokenizer(args):
    
    if args.model_type == 'llama':
        tokenizer = LLaMATokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_type == 'vit':
        tokenizer = ViTFeatureExtractor.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def build_gpt2_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_deberta_tokenizer(args):
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer_name)
    return tokenizer
    