import re
import datetime
import re

from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

login("sdfsd")  # Replace with your Hugging Face token
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
pipe = pipe.to("cuda")

# Ensure the tokenizer is fast and set add_prefix_space
if hasattr(pipe, "tokenizer"):
    if not pipe.tokenizer.is_fast:
        from transformers import AutoTokenizer
        pipe.tokenizer = AutoTokenizer.from_pretrained(pipe.tokenizer.name_or_path, use_fast=True)
    pipe.tokenizer.add_prefix_space = True

prompt = "old man with mustache, gray hair, white skin, only face"
safe_prompt = re.sub(r"[^a-zA-Z]", "_", prompt)

generationCount = 1
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    print("save image...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(timestamp + "_" + safe_prompt + ".png")
    print(i + 1, "of " + str(generationCount) + ": Generating image done for prompt:", prompt)

print("done")
