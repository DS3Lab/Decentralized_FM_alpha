import argparse
import os
import torch
from torch import autocast
from loguru import logger
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import random
from utils.dist_args_utils import *
from utils.coordinator_client import LocalCoordinatorClient
from utils.s3 import upload_file
from utils.local_coord import update_status


def main():

    parser = argparse.ArgumentParser(
        description='Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='test',
                        metavar='S', help='Job ID')

    add_global_coordinator_arguments(parser)
    add_lsf_coordinator_arguments(parser)
    args = parser.parse_args()
    print_arguments(args)
    update_status(args.job_id, "running")

    lsf_coordinator_client = LocalCoordinatorClient(
        "/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/working_dir/",
    )
    output_dir = os.path.join(
        "/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/new/working_dir/",
    )
    # lsf_coordinator_client.notify_inference_join()
    logger.info("Loading Stable Diffusion model...")
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

    logger.info("Stable Diffusion model loaded.")

    return_msg = lsf_coordinator_client.load_input_job_from_dfs(args.job_id)
    if return_msg is not None:
        logger.info(f"Received a new job. {return_msg}")

        job_requests = return_msg

        for job_request in job_requests:
            if isinstance(job_request['input'], str):
                text = [job_request['input']]
                num_return_sequences = [job_request['num_returns']]

            elif isinstance(job_request['input'], list):
                text = job_request['input']
                if isinstance(job_request['num_returns'], int):
                    num_return_sequences = [
                        job_request['num_returns']]*len(text)
                else:
                    num_return_sequences = job_request['num_returns']

            if len(text) != len(num_return_sequences):
                raise ValueError(
                    "The length of text and num_return_sequences (if given as a list) should be the same.")

            results = {}
            results['output'] = []
            with torch.no_grad():
                with autocast("cuda"):
                    img_results = []
                    generated_image_ids = []
                    for i in range(len(text)):
                        for j in range(num_return_sequences[i]):
                            image = pipe(text[i])["sample"][0]
                            # randomly generate a image id
                            image_id = random.randint(0, 1000000)
                            image.save(os.path.join(
                                output_dir, f"{image_id}.png"))
                            generated_image_ids.append(
                                os.path.join(output_dir, f"{image_id}.png"))
                            succ, img_id = upload_file(
                                os.path.join(output_dir, f"{image_id}.png"))
                            if succ:
                                img_results.append(
                                    "https://planetd.shift.ml/files/"+img_id)
                            else:
                                logger.error("Upload image failed")
                        results["output"].append(img_results)
                    update_status(
                        args.job_id,
                        "finished",
                        returned_payload=results
                    )
                    # clear cache
                    for image_id in generated_image_ids:
                        os.remove(image_id)


if __name__ == '__main__':
    main()