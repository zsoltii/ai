import os
import sys

MODEL = "stabilityai/sd-turbo"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diffusers import AutoPipelineForText2Image


from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.save_image import save_image


device = select_device()
log_available_gpus()

print("load model...")
model = AutoPipelineForText2Image.from_pretrained(MODEL)
print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
if device.type == 'cuda':
    model.enable_model_cpu_offload()
    model.enable_sequential_cpu_offload()
model.to(device)

prompt = "long hair women sitting on the beach, paint"

generationCount = 3
num_inference_steps = 10
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = model(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
    print("save paint...")
    save_image(prompt, image, MODEL, num_inference_steps)
    print(i + 1, "of " + str(generationCount) + ": Generating image done for prompt:", prompt)

print("done")
