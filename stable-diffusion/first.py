from diffusers import StableDiffusionPipeline
import torch
import re

model_id = "sd-legacy/stable-diffusion-v1-5"
device = "cpu"


pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

width = 1920
height = 1080

prompt = "home"
image = pipe(prompt, height=height, width=width).images[0]

safe_prompt = re.sub(r"[^a-zA-Z]", "_", prompt)
image.save(safe_prompt + "_" + str(width) + "x" + str(height) + ".png")
