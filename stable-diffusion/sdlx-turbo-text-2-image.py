import sys
import os

# MODEL = "stabilityai/sdxl-turbo"
MODEL = "thingthatis/sdxl-turbo"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers import AutoPipelineForText2Image
import torch

from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.save_image import save_image

log_available_gpus()
device = select_device()

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    # variant="fp16"
)
pipe.to(device)

prompt = "long hair women sitting on the beach, paint"

generationCount = 3
num_inference_steps = 10
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
    save_image(prompt, image, MODEL, num_inference_steps)
