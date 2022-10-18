import argparse
import time
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
import math
import numpy as np
import random


def get_huggingface_tokenizer_model(args, device):

    if args.model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b')
        model.config.eos_token_id = None
    elif args.model_name == 't0pp':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/T0pp')
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    elif args.model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2")
    elif args.model_name == 'gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    if args.fp16:
        model = model.half()
    model = model.to(device)
    return tokenizer, model


def pre_processing_texts(input_text, model_name):
    if model_name == 't5-11b':
        output_text = []
        for text in input_text:
            output_text.append(text+"<extra_id_0>")
        return output_text
    else:
        return input_text


def post_processing_text(input_text, output_text, model_name, query):
    print(f"<post_processing_text> input_text: {input_text}")
    print(f"<post_processing_text> output_text: {output_text}")
    stop_tokens = []
    for token in query.get('stop', []):
        if token != '':
            stop_tokens.append(token)
    print(f"<post_processing_text> stop_tokens: {stop_tokens}.")

    if query.get('max_tokens') == 0:
        return ""

    if model_name == 'gpt-j-6b':
        if not query.get('echo', False):
            text = output_text[len(input_text):]
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if query.get('echo', False):
                if text[len(input_text):].find(stop_token) != -1:
                    end_pos = min(text[len(input_text):].find(stop_token) + len(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    elif model_name == 'ul2' or model_name == 't0pp' or model_name == 't5-11b':
        if model_name == 't5-11b':
            input_text = input_text.replace("","")
        if query.get('echo', False):
            text = input_text+' '+output_text
        else:
            text = output_text
        end_pos = len(text)
        print(f"<post_processing_text>1 end_pos: {end_pos}.")
        for stop_token in stop_tokens:
            if query.get('echo', False):
                if text[len(input_text)+1:].find(stop_token) != -1:
                    end_pos = min(text[len(input_text)+1:].find(stop_token) + len(stop_token), end_pos)
            else:
                if text.find(stop_token) != -1:
                    end_pos = min(text.find(stop_token) + len(stop_token), end_pos)
            print(f"<post_processing_text>2 end_pos: {end_pos}.")
    else:
        assert False, "Model not supported yet."
    print(f"<post_processing_text> text: {text}, end_pos: {end_pos}")
    post_processed_text = text[:end_pos + 1]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def to_result(input_text, output_text, model_name, query):
    result = {}
    items = []
    for i in range(len(output_text)):
        item = {'choices': [], }
        print(f"<to_result> output{i}: {len(input_text[i])} / {len(output_text[i])}")
        choice = {
            "text": post_processing_text(input_text[i], output_text[i], model_name, query),
            "index": 0,
            "finish_reason": "length"
        }
        item['choices'].append(choice)
        items.append(item)
    result['inference_result'] = items
    return result


def main():
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--model-name', type=str, default='t5-11b', metavar='S',
                        help='trained model path')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S',
                        help='cuda-id (default:0)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='S',
                        help='batch-size for inference (default:8)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    args = parser.parse_args()
    print(args)
    local_cord_client = LocalCoordinatorClient(
        working_directory=args.working_directory,
        coordinator_url="http://localhost:5000/eth",
    )
    assert (torch.cuda.is_available())
    device = torch.device('cuda', args.cuda_id)
    try:
        tokenizer, model = get_huggingface_tokenizer_model(args, device)
        local_cord_client.update_status(args.job_id, "running")
    except Exception as e:
        print('Exception in model initialization inference:', e)
        error = traceback.format_exc()
        local_cord_client.update_status(args.job_id, "failed", returned_payload={"message": error})
        print(error)
        raise e

    try:
        while True:
            job_id = None
            raw_text = None
            try:
                instructions = local_cord_client.fetch_instructions(args.model_name, 0)
                last_instruction = instructions[-1]

                if last_instruction["message"] == "break":
                    logger.info("Received stop instruction.")
                    logger.info("# BREAK ")
                    break
                elif last_instruction["message"] == "continue":
                    logger.info(f"Received keep instruction. <{args.model_name}>")
                    sleep(1)
                elif last_instruction["message"] == "run":
                    fetched_tasks = [x for x in instructions
                                     if x["message"] == "run" and x['payload']['status'] == 'submitted']

                    if len(fetched_tasks) > 0:
                        instruction = fetched_tasks[0]
                        logger.info("Instruction:")
                        logger.info(str(instruction))
                        # TODO: we assume len(payload) is 1, right?
                        query = instruction['payload']['payload'][0]
                        if isinstance(query['prompt'], list):
                            raw_text = query['prompt']
                        elif isinstance(query['prompt'], str):
                            raw_text = [query['prompt']]
                        else:
                            print("wrong prompt format, it can only be str or list of str")
                            print(query['prompt'])

                        job_id = instruction['payload']['id']
                        print(f"Job <{job_id}> has been processed")

                        start_time = time.time()
                        
                        raw_text = pre_processing_texts(raw_text, args.model_name)

                        batch_size = min(len(raw_text), args.batch_size)
                        num_iter = math.ceil(len(raw_text) / batch_size)
                        answers = []
                        seed = query.get('seed', None)
                        if seed is not None:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            random.seed(seed)

                        for iter_i in range(num_iter):
                            current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
                            inputs = tokenizer(
                                current_raw_text,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                            )
                            inputs.to(device)
                            if query.get('temperature', 0.9) == 0:
                                outputs = model.generate(
                                    **inputs, do_sample=True, top_p=query.get('top_p', 0),
                                    temperature=1.0, top_k=1,
                                    max_new_tokens=query.get('max_tokens', 16),
                                    return_dict_in_generate=True,
                                    output_scores=True,  # return logit score
                                    output_hidden_states=True,  # return embeddings
                                )
                            else:
                                outputs = model.generate(
                                    **inputs, do_sample=True, top_p=query.get('top_p', 0),
                                    temperature=query.get('temperature', 0.9),
                                    max_new_tokens=query.get('max_tokens', 16),
                                    return_dict_in_generate=True,
                                    output_scores=True,  # return logit score
                                    output_hidden_states=True,  # return embeddings
                                )

                            current_output_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                            print(f"<Include_special_tokens>:", current_output_texts)
                            current_output_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                            answers.extend(current_output_texts)

                        end_time = time.time()
                        print(f"Job-{job_id} {args.model_name} Inference takes {end_time-start_time}s")
                        # print(f"outputs by hf model: {outputs}")
                        result = to_result(raw_text, answers, args.model_name, query)
                        return_payload = {
                            'request': query,
                            'result': result,
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
    main()
