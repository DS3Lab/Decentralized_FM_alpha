import os
import argparse
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modules.llama_modules import LLaMATokenizer, LLaMAForCausalLM, LLaMAConfig

from example_load_checkpoint_llama import *

config = LLaMAConfig.from_pretrained('/root/fm/models/_root_fm_models_llama-7b')
tokenizer = LLaMATokenizer.from_pretrained('/root/fm/models/_root_fm_models_llama-7b')

model = create_emtpy_llama(config).half()
load_decentralized_checkpoint(model, '/root/Decentralized_FM_alpha/model_checkpoints/llama-7b-alpaca-fix/checkpoint_800', n_stages=2, n_layer_per_stage=16)

save_path = '/root/fm/models/llama-7b-alpaca-fix-800'

config.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

model.save_pretrained(save_path)