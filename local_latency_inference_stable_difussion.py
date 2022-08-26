import io
import zipfile
import os, sys
from datetime import datetime
import json
import argparse
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from coordinator.global_coordinator.global_coordinator_client import GlobalCoordinatorClient


def main():
    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    lms = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=lms,
        use_auth_token=True
    ).to("cuda:0")

    add_global_coordinator_arguments(parser)
    global_coord_client = GlobalCoordinatorClient(args)
    return_msg = global_coord_client.get_request_cluster_coordinator()
    if return_msg['task_index'] == -1:
        time.sleep(10)
    else:


    text = ['a lovely dog with sunglasses']





    with torch.no_grad():
        with autocast("cuda"):
            image = pipe(text)["sample"][0]

    print(image)
    # image.save('./dog.jpeg', quality=95)

    return image


if __name__ == '__main__':
    main()