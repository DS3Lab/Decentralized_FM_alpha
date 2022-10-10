import argparse
import time
import os
import torch
import stat
import re
from functools import partial
from typing import List, Tuple
import torch.nn.functional as F
import numpy as np
from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.autoregressive_sampling import update_mems, get_masks_and_position_ids_default
from SwissArmyTransformer.mpu import vocab_parallel_cross_entropy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
# from SwissArmyTransformer.generation.sampling_strategies.base_strategy import top_k_logits
from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
import torch.distributed as dist
from time import sleep
import requests


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        #batch_size = logits.shape[0]
        # convert to 1D
        # logits = logits.view(-1).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        # logits = logits.view(1, -1).contiguous()

    return logits


class BaseStrategy:
    def __init__(self, batch_size, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None):
        self.batch_size = batch_size
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = np.zeros(self.batch_size, dtype=np.bool_)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        if dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>1 logits {logits.shape}, tokens {tokens.shape}, mems {mems.shape}")
        logits = logits.view(-1, logits.size(-1))
        if dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>2 logits {logits.shape}")
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if dist.get_rank() == 0:
            print(f"<BaseStrategy.forward>3 logits {logits.shape}")
        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                pred[i] = -1
            elif pred[i].item() in self.end_tokens:
                self._is_done[i] = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[:-1] + (1,))), dim=-1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = np.zeros(self.batch_size, dtype=np.bool_)
        return tokens, mems


def batch_filling_sequence(
        model,
        seqs,
        context_lengths,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        get_last_layer_embedding=False,
        **kw_args
        ):
    # print("<batch_filling_sequence> I am here 1")
    if dist.get_rank() == 0:
        print(f"<batch_filling_sequence> seqs: {seqs}")
    assert len(seqs.shape) == 2
    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape
   
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    # print("<batch_filling_sequence> I am here 2")
    output_embedding=None
    while counter < seqs.shape[1] - 1:
        if dist.get_rank() == 0:
            print(f"<batch_filling_sequence> counter:{counter}/{seqs.shape[1] - 1}")
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        
        output_embedding_flag = get_last_layer_embedding and counter == context_length - 1
        logits, *output_per_layers = model(
            tokens[:, index:],
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
            output_hidden_states=output_embedding_flag
        )
        if output_embedding_flag:
            output_embedding = output_per_layers[-1]['hidden_states']
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        if counter == context_length - 1:
            logits = logits[torch.arange(batch_size), context_lengths - 1]
        else:
            logits = logits[:, -1]
        counter += 1
        index = counter
        # if torch.distributed.get_rank() == 0:
        #     print(f"counter: {counter}: logits: {logits.float().abs().mean()}")
        # sampling
        logits = logits.reshape(batch_size, num_beams, -1)
        tokens = tokens.reshape(batch_size, num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size, num_beams, mems.shape[-2], mems.shape[-1])
        tokens, mems = strategy.forward(logits, tokens, mems)
        if len(tokens.shape) == 3 and num_beams == 1:
            num_beams = tokens.shape[1]
            position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1, -1, -1).reshape(
                batch_size * num_beams, *attention_mask_shape)
        if strategy.is_done:
            break
    if get_last_layer_embedding:
        tokens, mems = strategy.finalize(tokens, mems)
        return tokens, mems, output_embedding
    else:
        return strategy.finalize(tokens, mems)


def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group("BMInf")

    group.add_argument("--bminf", action="store_true", help="Use BMInf to support low resource evaluation")
    group.add_argument("--bminf-memory-limit", type=int, default=20, help="Max memory for model per GPU (in GB)")
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group("Quantization")

    group.add_argument("--quantization-bit-width", type=int, default=None)
    group.add_argument("--from-quantized-checkpoint", action="store_true", help="Loading from a quantized checkpoint")


def foo_port_add_coordinator_args(parser):
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/root/fm/working_dir', metavar='S',
                        help='Working_dir for file cache.')


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    foo_port_add_coordinator_args(parser)
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    return args


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    # Initialize model
    model = GLM130B(args).half()

    # Load checkpoint
    torch.distributed.barrier()
    start = time.time()
    load_checkpoint(model, args)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"> Checkpoint loaded in {time.time() - start:.1f}s")

    if args.bminf:
        import bminf

        if torch.distributed.get_rank() == 0:
            print(f"> BMInf activated, memory limit: {args.bminf_memory_limit} GB")
        with torch.cuda.device(args.device):
            model = bminf.wrapper(model, quantization=False, memory_limit=args.bminf_memory_limit << 30)
    else:
        model = model.to(args.device)

    torch.cuda.empty_cache()
    model.eval()

    # generate rotary embedding cache
    original_parallel_output = model.transformer.parallel_output
    model.transformer.parallel_output = True
    with torch.no_grad():
        _, *_ = model(
            torch.ones(1, args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64),
            torch.arange(args.max_sequence_length, device=torch.cuda.current_device(), dtype=torch.int64).view(1, -1),
            torch.randn(
                1,
                1,
                args.max_sequence_length,
                args.max_sequence_length,
                device=torch.cuda.current_device(),
            )
            < 0.5,
        )
    model.transformer.parallel_output = original_parallel_output
    torch.distributed.barrier()

    return model, tokenizer


def add_generation_specific_args(parser):
    parser.add_argument("--sampling-strategy", type=str, default="BaseStrategy", help="Type of sampling strategy.")
    parser.add_argument("--min-gen-length", type=int, default=0, help="The minimum length each blank should generate.")
    parser.add_argument(
        "--print-all-beams", action="store_true", help="Print all output generated by beam search strategy."
    )


def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_masks_and_position_ids(seq, mask_position, max_gen_length, gmask=False):
    context_length = seq.shape[1]
    tokens = torch.nn.functional.pad(seq, (0, max_gen_length), mode='constant', value=-1)
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1 :] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def get_masks_and_position_ids_batch(seqs, mask_position, max_gen_length, pad_pos, gmask=False):
    batch_size = seqs.shape[0]
    context_length = seqs.shape[1]
    tokens = torch.nn.functional.pad(seqs, (0, max_gen_length), mode='constant', value=-1)
    # TODO This might be wrong, double check.
    attention_mask = torch.ones((batch_size, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    
    for i in range(batch_size):
        attention_mask[i, :, 0:pad_pos[i]] = 0
        attention_mask[i, :, pad_pos[i]: context_length - 1] = 1
    
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    
    position_ids = torch.zeros((batch_size,tokens.shape[-1]), dtype=torch.long, device=tokens.device)
    
    for i in range(batch_size):
        position_ids[i] = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device) - pad_pos[i]
        position_ids[i, 0:pad_pos[i]] = 0
        
    if not gmask:
        position_ids[:, context_length - 1:] = mask_position

    if dist.get_rank() == 0:
        print(f"<get_masks_and_position_ids_batch> tokens: {tokens}")
        print(f"<get_masks_and_position_ids_batch> attention_mask: {attention_mask}")
        print(f"<get_masks_and_position_ids_batch> position_ids: {position_ids}")


    return tokens, attention_mask, position_ids


def fill_blanks(raw_text: str, model, tokenizer, strategy, config=None):
    # add MASK
    generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"
    use_gmask = "[MASK]" not in raw_text
    last_layer_embedding = None

    mask_pattern = r"\[g?MASK\]"
    text_list = re.split(mask_pattern, raw_text)
    pattern_list = re.compile(mask_pattern).findall(raw_text)
    seq = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        sub_text = text_list[i]
        seq.extend(tokenizer.tokenize(sub_text))
        seq.append(tokenizer.get_command(pattern))

    seq.extend(tokenizer.tokenize(text_list[-1]))

    if "MASK]" not in raw_text:
        seq += [tokenizer.get_command(generation_mask)]
        raw_text += " " + generation_mask
    if not raw_text.endswith("MASK]"):
        seq = seq + [tokenizer.get_command("eos")]
    if mpu.get_model_parallel_rank() == 0:
        print("\nInput: {}\n".format(raw_text))
    if config is None and len(seq) > args.max_sequence_length:
        raise ValueError("text too long.")

    # generation
    is_english = isEnglish(raw_text)
    output_list = [seq]
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1
    last_pos, answers, answers_with_style, blanks = (
        [0] * num_output,
        ["" for _ in range(num_output)],
        ["" for _ in range(num_output)],
        [[] for _ in range(num_output)],
    )
    # print("<fill_blanks> I am here 1")
    # continually detect the first mark position
    foo_counter = 0
    while True:
        if dist.get_rank() == 0:
            print(f"<batch_filling_sequence> counter:{foo_counter}")
            foo_counter += 1
            
        seq = output_list[0]
        # detect mask position
        mask_token = tokenizer.get_command(generation_mask)
        if mask_token not in seq:
            break
        mask_position = seq.index(mask_token)

        output_list = []

        input_seq = torch.cuda.LongTensor(
            [seq + [tokenizer.get_command("sop")]],
            device=args.device,
        )
        # print("<fill_blanks> I am here 2")
        if config is not None and config['prompt_embedding']:
            get_last_layer_embedding = True
        else:
            get_last_layer_embedding = False

        if get_last_layer_embedding:
            output, _, last_layer_embedding = batch_filling_sequence(
                model,
                input_seq,
                torch.cuda.LongTensor([input_seq.shape[-1]], device=args.device),
                strategy=strategy,
                get_masks_and_position_ids=partial(
                    get_masks_and_position_ids,
                    mask_position=mask_position,
                    max_gen_length=config['max_tokens'] if config else args.out_seq_length - input_seq.shape[-1],
                    gmask=use_gmask,
                ),
                get_last_layer_embedding=get_last_layer_embedding
            )
        else:
            output, _ = batch_filling_sequence(
                model,
                input_seq,
                torch.cuda.LongTensor([input_seq.shape[-1]], device=args.device),
                strategy=strategy,
                get_masks_and_position_ids=partial(
                    get_masks_and_position_ids,
                    mask_position=mask_position,
                    max_gen_length=config['max_tokens'] if config else args.out_seq_length - input_seq.shape[-1],
                    gmask=use_gmask,
                ),
                get_last_layer_embedding=get_last_layer_embedding
            )
        # print("<fill_blanks> I am here 3")
        if isinstance(output, torch.Tensor):  # different strategies
            output = output.tolist()
        output = output[0]  # batch_size = 1
        output_list.extend(output)

        # clip -1s and fill back generated things into seq
        for i in range(len(output_list)):
            output = output_list[i].tolist() if isinstance(output_list[i], torch.Tensor) else output_list[i]
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in strategy.end_tokens:
                unfinished -= 1
            bog = output.index(tokenizer.get_command("sop"))

            prefix = tokenizer.detokenize(output[last_pos[i] : mask_position])
            blank = tokenizer.detokenize(output[bog + 1 : unfinished])
            answers_with_style[i] += (
                prefix
                + (" " if is_english else "")
                + ("\033[4m" if use_gmask else "\x1b[0;32m\033[4m")
                + blank
                + ("\033[0m" if use_gmask else "\033[0m\x1b[0m")
                + (" " if is_english else "")
            )
            blanks[i].append(blank)
            last_pos[i] = mask_position + unfinished - (bog + 1)
            output_list[i] = output[:mask_position] + output[bog + 1 : unfinished] + output[mask_position + 1 : bog]

        # print("<fill_blanks> I am here 4")

    for i, output in enumerate(output_list):
        if output[-1] == tokenizer.get_command("eos"):
            output = output[:-1]
        answers_with_style[i] += tokenizer.detokenize(output[last_pos[i] :])
        if dist.get_rank() == 0:
            print(f"<fill_blanks> output: {output}")
        answers[i] = tokenizer.detokenize(output)

    # print("<fill_blanks> I am here 5")
    return answers, answers_with_style, blanks, last_layer_embedding



def fill_blanks_efficient(raw_texts: str, model, tokenizer, strategy, config=None):
    seqs = []
    generation_mask = "[gMASK]"
    use_gmask = True
    last_layer_embedding = None
    for raw_text in raw_texts:
        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(tokenizer.tokenize(sub_text))
            seq.append(tokenizer.get_command(pattern))

        seq.extend(tokenizer.tokenize(text_list[-1]))

        if "MASK]" not in raw_text:
            seq += [tokenizer.get_command(generation_mask)]
            raw_text += " " + generation_mask
        if not raw_text.endswith("MASK]"):
            seq = seq + [tokenizer.get_command("eos")]
        if mpu.get_model_parallel_rank() == 0:
            print("\nInput: {}\n".format(raw_text))
        if config is None and len(seq) > args.max_sequence_length:
            raise ValueError("text too long.")
        seqs.append(seq)

    # for ii in range(len(seqs)-1):
    #    assert len(seqs[ii]) == len(seqs[ii+1])

    # generation
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1

    # print("<fill_blanks> I am here 1")
    # continually detect the first mark position

    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> seqs :{seqs}")

    # detect mask position
    mask_token = tokenizer.get_command(generation_mask)
    
    batch_size = len(seqs)
    context_length = 0
    for seq in seqs:
        if len(seq) > context_length:
            context_length = len(seq)
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> batch_size: {batch_size}, context_length: {context_length}")
    
    padding_pos = []
    for seq in seqs:
        padding_pos.append(context_length - len(seq))
        
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> padding_pos {padding_pos}")
    
    input_seqs = torch.cuda.LongTensor(
        [[0] * (context_length - len(seq)) + seq + [tokenizer.get_command("sop")] for seq in seqs],
        device=args.device,
    )
    
    
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> input_seqs.shape :{input_seqs.shape}, input_seqs: {input_seqs}")
    mask_position = context_length - 1    
    
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> input_seqs {input_seqs}")
    
    # print("<fill_blanks> I am here 2")
    if config is not None and config['prompt_embedding']:
        get_last_layer_embedding = True
    else:
        get_last_layer_embedding = False

    if get_last_layer_embedding:
        outputs, _, last_layer_embedding = batch_filling_sequence(
            model,
            input_seqs,
            torch.cuda.LongTensor([input_seqs.shape[-1] for _ in range(input_seqs.shape[0])], device=args.device),
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids_batch,
                mask_position=mask_position,
                max_gen_length=config['max_tokens'] if config else args.out_seq_length - input_seqs.shape[-1],
                pad_pos = padding_pos,
                gmask=use_gmask,
            ),
            get_last_layer_embedding=get_last_layer_embedding
        )
    else:
        outputs, _ = batch_filling_sequence(
            model,
            input_seqs,
            torch.cuda.LongTensor([input_seqs.shape[-1] for _ in range(input_seqs.shape[0])], device=args.device),
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids_batch,
                mask_position=mask_position,
                max_gen_length=config['max_tokens'] if config else args.out_seq_length - input_seqs.shape[-1],
                pad_pos = padding_pos,
                gmask=use_gmask,
            ),
            get_last_layer_embedding=get_last_layer_embedding
        )
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> outputs:{outputs.shape}")
    answers = []
    for i in range(outputs.shape[0]):
        answers_per_seq = []
        for j in range(num_output):
            output = outputs[i][j]
            if output[-1] == tokenizer.get_command("eos"):
                output = output[:-1]
            if dist.get_rank() == 0:
                print(f"<fill_blanks_efficient> output :{output.shape}")
            answers_per_seq.append(tokenizer.detokenize(output[padding_pos[i]:].tolist()))
        answers.append(answers_per_seq)
    if dist.get_rank() == 0:
        print(f"<fill_blanks_efficient> answers: {answers}")
    
    if last_layer_embedding is not None:
        last_layer_embedding = torch.transpose(last_layer_embedding, 0, 1)
        last_layer_embeddings = []
        for i in range(batch_size):
            current_sample_embedding = last_layer_embedding[i, padding_pos[i]:, :]
            if dist.get_rank() == 0:
                print(f"<fill_blanks_efficient> current_sample_embedding_{i} .shape: {current_sample_embedding.shape}")
            last_layer_embeddings.append(current_sample_embedding)
    else:
        last_layer_embeddings = None
    
    return answers, last_layer_embeddings


def to_result(output, query, prompt_str_length, last_layer_embedding, job_id=None, working_directory=None):
    print(f"<to_result> output: {output}")
    # TODO, Lots of missing attributes here!!!!
    if len(output) == 1:
        item = {'choices': [], }
        if query.get('max_tokens') == 0:
            text = ""
        elif query.get('echo', False):
            text = output[0][0].replace("[[gMASK]][sop]", " ")  
        else:
            text = output[0][0][prompt_str_length[0]:].replace("[[gMASK]][sop]", "")
        choice = {
            "text": text,
            "index": 0,
            "finish_reason": "length"
        }
        if last_layer_embedding is not None:
            print(f"serialize last layer embedding, shape {last_layer_embedding} ")
            tensor_filename = working_directory+'/'+job_id+'_embedding.pt'
            torch.save(last_layer_embedding, tensor_filename)
            with open(tensor_filename, "rb") as fp:
                files = {"file": fp}
                res = requests.post("https://planetd.shift.ml/file", files=files).json()
                choice['embedding'] = res["filename"]
                os.remove(tensor_filename)
        item['choices'].append(choice)
        return item
    else:
        result = {}
        items = []
        if last_layer_embedding is not None:
            #last_layer_embedding = torch.transpose(last_layer_embedding, 0, 1)
            print(f"serialize last layer embeddings {last_layer_embedding} ")
            tensor_filename = working_directory+'/'+job_id+'_embedding.pt'
            torch.save(last_layer_embedding, tensor_filename)
            with open(tensor_filename, "rb") as fp:
                files = {"file": fp}
                res = requests.post("https://planetd.shift.ml/file", files=files).json()
                result['embedding'] = res["filename"]
                os.remove(tensor_filename)
        
        for i in range(len(output)):
            item = {'choices': [], }
            print(f"<to_result> output{i}: {prompt_str_length[i]} / {len(output[i][0])}")
            if query.get('max_tokens') == 0:
                text = ""
            elif query.get('echo', False):
                text = output[i][0].replace("[[gMASK]][sop]", " ")  
            else:
                text = output[i][0][prompt_str_length[i]:].replace("[[gMASK]][sop]", "")
            choice = {
                "text": text,
                "index": 0,
                "finish_reason": "length"
            }
            item['choices'].append(choice)
            items.append(item)
        result['inference_result'] = items
        return result
    


def main(args):
    if dist.get_rank() == 0:
        print(args)

    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    try:
        model, tokenizer = initialize_model_and_tokenizer(args)
        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    try:
        while True:
            try:
                has_work = False
                raw_text = ""
                config = {}
                if dist.get_rank() == 0:
                    instructions = local_cord_client.fetch_instructions('glm', 0)
                    last_instruction = instructions[-1]

                    if last_instruction["message"] == "break":
                        logger.info("Received stop instruction. <GLM>")
                        logger.info("# BREAK ")
                        break
                    elif last_instruction["message"] == "continue":
                        logger.info("Received keep instruction. <GLM>")
                        sleep(1)
                        has_work = False
                    elif last_instruction["message"] == "run":
                        fetched_tasks = [x for x in instructions
                                         if x["message"] == "run" and x['payload']['status'] == 'submitted']

                        if len(fetched_tasks) > 0:
                            instruction = fetched_tasks[0]
                            logger.info("Instruction:")
                            logger.info(str(instruction))
                            job_id = instruction['payload']['id']
                            print(f"Job <{job_id}> has been batched")
                            
                            # TODO: we assume len(payload) is 1, right?
                            query = instruction['payload']['payload'][0]
                            if isinstance(query['prompt'], list):
                                raw_text = query['prompt']
                                for i in range(len(raw_text)):
                                    raw_text[i] = raw_text[i].strip()
                            elif isinstance(query['prompt'], str):
                                raw_text = query['prompt']
                                raw_text = [raw_text.strip()]
                            else:
                                print("wrong prompt format, it can only be str or list of str")
                                print(query['prompt'])
                            
                            config = {
                                'temperature': query.get('temperature', 0.9),
                                # 'top_k': query.get('top_k', 1),
                                'top_p': query.get('top_p', 0),
                                'max_tokens': query.get('max_tokens',10) if query.get('max_tokens',10) > 0 else 1,
                                'prompt_embedding': query.get('prompt_embedding', False)
                            }
                            has_work = True
                        else:
                            has_work = False

                dist.barrier()
                if dist.get_rank() == 0:
                    dist.broadcast_object_list([raw_text, config, has_work])
                else:
                    info = [raw_text, config, has_work]
                    torch.distributed.broadcast_object_list(info)
                    raw_text, config, has_work = info
                dist.barrier()

                if has_work:
                    print(f"Rank-<{dist.get_rank()}> join inference.")
                    start_time = time.time()
                    # strategy = BaseStrategy(batch_size=1, temperature=args.temperature, top_k=args.top_k,
                    #                        top_p=args.top_p, end_tokens=end_tokens)
                    # Followed Jue's suggestion for temperature
                    if config['temperature'] == 0:
                        strategy = BaseStrategy(batch_size=len(raw_text), temperature=1, top_k=1,
                                                top_p=config['top_p'], end_tokens=end_tokens)
                    else:
                        strategy = BaseStrategy(batch_size=len(raw_text), temperature=config['temperature'], top_k=args.top_k,
                                                top_p=config['top_p'], end_tokens=end_tokens)

                    # TODO change config to our config, to make it work desired seq length.
                    # answers, answers_with_style, blanks, last_layer_embedding = \
                    #    fill_blanks(raw_text[0], model, tokenizer, strategy, config)
                    answers, last_layer_embedding = fill_blanks_efficient(raw_text, model, tokenizer, strategy, config)
                    end_time = time.time()
                    # print(f"Rank-<{dist.get_rank()}>: answer:")
                    # print(answers)
                    if dist.get_rank() == 0:
                        print(f"Job-{job_id} GLM Inference takes {end_time-start_time}s")
                        prompt_str_lengths = []
                        for text in raw_text:
                            prompt_str_lengths.append(len(text))
                        result = to_result(answers, query, prompt_str_lengths, last_layer_embedding, job_id=job_id, 
                                           working_directory=args.working_directory)
                        return_payload = {
                            'request': query,
                            'result': result,
                            'raw_compute_time': end_time-start_time
                        }
                        # local_cord_client.update_status(
                        local_cord_client.update_status_global_coordinator(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )
                        local_cord_client.update_status(job_id, "finished", returned_payload={})

            except Exception as e:
                error = traceback.format_exc()
                if dist.get_rank() == 0:
                    local_cord_client.update_status(
                        job_id,
                        "failed",
                        returned_payload={"message": error}
                    )
                print(error)
                raise e

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
