import torch
import torch.nn as nn

from modules.llama_modules import LLaMAForCausalLM

from modules.llama_modules import LLaMAConfig, LLaMATokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_llama(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = LLaMAForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=16, ):
    input_path = checkpoint_path

    n_layers = len(model.model.layers)
    assert n_stages * n_layer_per_stage >= len(model.model.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.model.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.model.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                if i*n_layer_per_stage + j == n_layers:
                    break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)
            else:
                j += 1

            _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.model.norm.weight.data[:] = _tmp['norm.weight']
            if 'norm.bias' in _tmp:
                model.model.norm.bias.data[:] = _tmp['norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


if __name__ == '__main__':

    config = LLaMAConfig.from_pretrained('/root/fm/models/_root_fm_models_llama-7b')
    tokenizer = LLaMATokenizer.from_pretrained('/root/fm/models/_root_fm_models_llama-7b')
    model = create_emtpy_llama(config)
    load_decentralized_checkpoint(model, '/root/Decentralized_FM_alpha/model_checkpoints/llama-7b-alpaca-fix/checkpoint_300', n_stages=2, n_layer_per_stage=16)

    # test on cpu
    ret = model.generate(**tokenizer('you are not', return_tensors='pt'), max_new_tokens=4)

    print(tokenizer.batch_decode(ret))
