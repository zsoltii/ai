from diffusers import StableDiffusionPipeline
import torch
import re
import datetime

model_id = "sd-legacy/stable-diffusion-v1-5"
# model_id = "CompVis/stable-diffusion-v1-1"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
#
# width = 512*2
# height = 512*2

prompt = "one lion in a field, high quality, detailed, realistic, cinematic lighting"

safe_prompt = re.sub(r"[^a-zA-Z]", "_", prompt)

seed = 42
generator = torch.Generator(device="cpu").manual_seed(seed)

generationCount = 10
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(prompt, generator=generator).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(timestamp + "_" + safe_prompt + ".png")
#
# image = pipe(prompt, height=height, width=width).images[0]
# image.save(safe_prompt + "_" + str(width) + "x" + str(height) + ".png")
