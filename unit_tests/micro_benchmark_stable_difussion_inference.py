import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler



def main():
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

    text = ['a lovely dog with sunglasses']

    with torch.no_grad():
        with autocast("cuda"):
            image = pipe(text)["sample"][0]

    image.save('./dog.jpeg', quality=95)

    return image


if __name__ == '__main__':
    main()