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
from SwissArmyTransformer.generation.sampling_strategies.base_strategy import top_k_logits
from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
import torch.distributed as dist
from time import sleep


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
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        logits = logits.view(-1, logits.size(-1))
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504

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
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)
        return tokens, mems


def batch_filling_sequence(
        model,
        seqs,
        context_lengths,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        **kw_args
        ):
    # print("<batch_filling_sequence> I am here 1")
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
    while counter < seqs.shape[1] - 1:
        if dist.get_rank() == 0:
            print(f"<batch_filling_sequence> counter:{counter}/{seqs.shape[1] - 1}")
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        logits, *output_per_layers = model(
            tokens[:, index:],
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
            **kw_args
        )
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
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')


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


def fill_blanks(raw_text: str, model, tokenizer, strategy, config=None) -> Tuple[List[str], List[str], List[List[str]]]:
    # add MASK
    generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"
    use_gmask = "[MASK]" not in raw_text

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
    while True:
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
        answers[i] = tokenizer.detokenize(output)

    # print("<fill_blanks> I am here 5")
    return answers, answers_with_style, blanks


def to_result(output):
    # TODO, Lots of missing attributes here!!!!
    item = {'choices': [], }
    choice = {
        "text": (output[0]),
        "index": 0,
        "finish_reason": "length",
    }
    item['choices'].append(choice)
    return item


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
                        logger.info("Received stop instruction.")
                        logger.info("# BREAK ")
                        break
                    elif last_instruction["message"] == "continue":
                        logger.info("Received keep instruction.")
                        has_work = False
                    elif last_instruction["message"] == "run":
                        fetched_tasks = [x for x in instructions
                                         if x["message"] == "run" and x['payload']['status'] == 'submitted']

                        if len(fetched_tasks) > 0:
                            instruction = fetched_tasks[0]
                            logger.info("Instruction:")
                            logger.info(str(instruction))
                            # TODO: we assume len(payload) is 1, right?
                            query = instruction['payload']['payload'][0]
                            raw_text = query['prompt']
                            raw_text = raw_text.strip()
                            job_id = instruction['payload']['id']
                            print(f"Job <{job_id}> has been batched")
                            config = {
                                'temperature': query.get('temperature', 0.9),
                                'top_k': query.get('top_k', 1),
                                'top_p': query.get('top_p', 0),
                                'max_tokens': query.get('max_tokens',10)
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
                    # TODO some of our input config is illegal for GLM, need a fix!
                    strategy = BaseStrategy(batch_size=1, temperature=config['temperature'], top_k=config['top_k'],
                                            top_p=args.top_p, end_tokens=end_tokens)
                    # TODO change config to our config, to make it work desired seq length.
                    answers, answers_with_style, blanks = fill_blanks(raw_text, model, tokenizer, strategy, config)
                    end_time = time.time()
                    # print(f"Rank-<{dist.get_rank()}>: answer:")
                    # print(answers)
                    if dist.get_rank() == 0:
                        print(f"Job-{job_id} GLM Inference takes {end_time-start_time}s")
                        result = to_result(answers)
                        return_payload = {
                            'request': query,
                            'result': result,
                        }
                        local_cord_client.update_status(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )

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
            sleep(1)

    except Exception as e:
        print('Exception in latency inference:', e)


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
