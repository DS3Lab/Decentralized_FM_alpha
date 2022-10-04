import argparse
import time
from coordinator.coordinator_client import LocalCoordinatorClient
import traceback
from loguru import logger
import torch.distributed as dist
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch


def get_huggingface_tokenizer_model(args, device):

    if args.model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b')
        model = T5ForConditionalGeneration.from_pretrained('t5-11b')
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


def main():
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--model-name', type=str, default='t5-11b', metavar='S',
                        help='trained model path')
    parser.add_argument('cuda-id', type=int, default=0, metavar='S',
                        help='--cuda-id (default:0)')
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
        tokenizer, model = get_huggingface_tokenizer_model(args.model_name, device)
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
                    logger.info("Received keep instruction.")
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
                            raw_text = query['prompt'][0]
                        elif isinstance(query['prompt'], str):
                            raw_text = query['prompt']
                        else:
                            print("wrong prompt format, it can only be str or list of str")
                            print(query['prompt'])

                        job_id = instruction['payload']['id']
                        print(f"Job <{job_id}> has been processed")

                        start_time = time.time()
                        prompt = [raw_text]

                        inputs = tokenizer(
                            prompt,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )

                        outputs = model.generate(
                            **inputs, do_sample=True, top_p=query.get('top_p', 0),
                            temperature=query.get('temperature', 0.9),
                            max_new_tokens=query.get('max_tokens', 16),
                            return_dict_in_generate=True,
                            output_scores=True,  # return logit score
                            output_hidden_states=True,  # return embeddings
                        )

                        texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

                        print(texts)

                        end_time = time.time()
                        print(f"Job-{job_id} GLM Inference takes {end_time-start_time}s")
                        print(f"outputs by hf model: {outputs}")
                        # result = to_result(answers, query, len(raw_text))
                        '''
                        return_payload = {
                            'request': query,
                            'result': result,
                        }
                        local_cord_client.update_status(
                        local_cord_client.update_status_global_coordinator(
                            job_id,
                            "finished",
                            returned_payload=return_payload
                        )
                        local_cord_client.update_status(job_id, "finished", returned_payload={})
                        '''

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
    main()



