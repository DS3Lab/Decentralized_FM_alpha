from datetime import datetime
import time
import base64
from io import BytesIO
import argparse
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from coordinator.lsf.lsf_coordinate_client import CoordinatorInferenceClient
from utils.dist_args_utils import *


def main():

    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    parser.add_argument('--infer-data', type=str, default='foo', metavar='S',
                        help='data path')
    add_global_coordinator_arguments(parser)
    add_lsf_coordinator_arguments(parser)
    args = parser.parse_args()
    print_arguments(args)

    lsf_coordinator_client = CoordinatorInferenceClient(args, 'stable_diffusion')
    lsf_coordinator_client.notify_inference_join()

    lms = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=lms,
        use_auth_token=True,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda:0")

    print("Load Stable Diffusion Model is done.")

    lsf_coordinator_client.notify_inference_heartbeat()
    last_timestamp = time.time()
    time.sleep(10)

    while True:
        current_timestamp = time.time()
        if current_timestamp - last_timestamp >= args.heartbeats_timelimit:
            lsf_coordinator_client.notify_inference_heartbeat()
            last_timestamp = current_timestamp
            time.sleep(10)

        return_msg = lsf_coordinator_client.load_input_job_from_dfs()

        if return_msg is not None:
            print(f"Handel request: <{return_msg['_id']}>")

            job_request = return_msg

            num_return_sequences = job_request['task_api']['parameters']['num_return_sequences']
            text = [job_request['task_api']['inputs']]
            with torch.no_grad():
                with autocast("cuda"):
                    img_results = []
                    for i in range(num_return_sequences):
                        image = pipe(text)["sample"][0]
                        # print(image)
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_encode = base64.b64encode(buffered.getvalue())
                        img_str = img_encode.decode()  # image stored in base64 string.
                        img_results.append(img_str)
                    # print(img_str)
                    job_request['task_api']['outputs'] = img_results
                    lsf_coordinator_client.save_output_job_to_dfs(job_request)


if __name__ == '__main__':
    main()