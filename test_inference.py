import torch
import torch.nn as nn

from transformers import GPTJForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_gptj(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTJForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.transformer.h)
    assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.transformer.wte.weight.data[:] = _tmp['wte.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.transformer.h[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i*n_layer_per_stage + j].load_state_dict(_tmp)

            _tmp = {k[len(f"{n_layer_per_stage}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.transformer.ln_f.weight.data[:] = _tmp['ln_f.weight']
            model.transformer.ln_f.bias.data[:] = _tmp['ln_f.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model

if __name__ == '__main__':
    import os
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    model = create_emtpy_gptj(config)
    finetune_id = os.environ.get("FINETUNE_ID")
    # get all checkpoints
    checkpoints = os.environ.get("TOTAL_STEPS")
    finetune_path = os.path.join("model_checkpoints", finetune_id)
    load_decentralized_checkpoint(model, finetune_path, n_stages=2, n_layer_per_stage=14)

    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=["GPTJBlock"],
        dtype='float16',
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["GPTJBlock"],
        dtype='float16'
    )

    model = dispatch_model(model, device_map=device_map)

    ret = model.generate(**tokenizer('you are not', return_tensors='pt'), max_new_tokens=4)
    print(ret)
    print(tokenizer.batch_decode(ret))
    """
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
    """
