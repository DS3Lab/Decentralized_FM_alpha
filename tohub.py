import os
from transformers import OPTForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

from transformers import OPTForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights


def create_emtpy_opt(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = OPTForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.model.decoder.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.model.decoder.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']
            model.model.decoder.embed_positions.weight.data[:] = _tmp['embed_positions.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.model.decoder.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if 'lm_head.weight' in _tmp:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.decoder.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)
            else:
                _tmp = {k[len(f"{n_layer_per_stage}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
                
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.model.decoder.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.model.decoder.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.model.decoder.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


if __name__ == '__main__':

    config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
    model = create_emtpy_opt(config)
    
    finetune_id = os.environ.get("FINETUNE_ID")
    checkpoints = os.environ.get("TOTAL_STEPS")
    finetune_path = os.path.join("model_checkpoints", finetune_id)
    load_decentralized_checkpoint(model, finetune_path, n_stages=2, n_layer_per_stage=12)
    tokenizer = AutoTokenizer.from_pretrained(finetune_path)
    tokenizer.push_to_hub(
        repo_id=f"{finetune_id}",
        repo_path_or_name=f"./model_checkpoints/{finetune_id}",
        repo_url=f"https://huggingface.co/xzyao/{finetune_id}",
        use_auth_token=True,
    )
    model.push_to_hub(
        repo_id=f"{finetune_id}",
        repo_path_or_name=f"./model_checkpoints/{finetune_id}",
        repo_url=f"https://huggingface.co/xzyao/{finetune_id}",
        use_auth_token=True,
    )