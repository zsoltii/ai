import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers import DiffusionPipeline
import torch

from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.save_image import save_image

device = select_device()
log_available_gpus()

modelName = "stabilityai/stable-diffusion-xl-base-1.0"

if(device.type == 'cpu'):
    pipe = DiffusionPipeline.from_pretrained(modelName, use_safetensors=True)
else:
    pipe = DiffusionPipeline.from_pretrained(modelName, dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
pipe.to(device)

prompt = "long hair women sitting on the beach, paint"

generationCount = 3
num_inference_steps = 50
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
    save_image(prompt, image, modelName, num_inference_steps)