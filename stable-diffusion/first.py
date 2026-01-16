import os
import sys

from triton.language import dtype

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers import StableDiffusionPipeline
import torch
from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.save_image import save_image

log_available_gpus()
device = select_device()

model_id = "sd-legacy/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.bfloat16)
# if device.type == 'cuda':
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
print(f'Pipeline loaded to device: {device.type}')
pipe = pipe.to(device)

prompt = "long hair women sitting on the beach, paint"

generationCount = 3
num_inference_steps = 10
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    save_image(prompt, image, model_id, num_inference_steps)
