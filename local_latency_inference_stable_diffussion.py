from datetime import datetime
import time
import base64
from io import BytesIO
import argparse
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from coordinator.global_coordinator.global_coordinator_client import GlobalCoordinatorClient
from coordinator.lsf.lsf_coordinate_client import CoordinatorInferenceClient
from utils.dist_args_utils import *


def main():
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

    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    parser.add_argument('--infer-data', type=str, default='foo', metavar='S',
                        help='data path')
    add_global_coordinator_arguments(parser)
    add_lsf_coordinator_arguments(parser)
    args = parser.parse_args()
    print_arguments(args)

    lsf_coordinator_client = CoordinatorInferenceClient(args)
    lsf_coordinator_client.notify_inference_join()

    global_coord_client = GlobalCoordinatorClient(args)

    while True:
        return_msg = global_coord_client.get_request_cluster_coordinator(model_name='stable_diffusion',
                                                                         task_type='image_generation')
        # print("<<<<<<<<<<<<<<Return_msg Dict>>>>>>>>>>>>")
        if return_msg:
            print(f"Handel request: <{return_msg['_id']}>")

        if return_msg is None:
            time.sleep(10)
        else:
            num_return_sequences = return_msg['task_api']['parameters']['num_return_sequences']
            text = [return_msg['task_api']['inputs']]
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
                    global_coord_client.put_request_cluster_coordinator(return_msg, img_results)


if __name__ == '__main__':
    main()