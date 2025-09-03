import sys
import os

MODEL = "sd-legacy/stable-diffusion-inpainting"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from util.select_device import select_device
from util.log_available_gpus import log_available_gpus
from util.save_image import save_image

log_available_gpus()
device = select_device()

print("load model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
)

if device.type == 'cuda':
    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
print(f'Pipeline loaded to device: {device.type}')
pipe = pipe.to(device)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

# Load images as PIL images
print("load image...")
image = Image.open("images/image.png").convert("RGB")
print("load mask...")
mask_image = Image.open("images/mask.png").convert("L")  # L mode for mask

print("generate paint for prompt: " + prompt)
# The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
print("save paint...")
save_image(prompt, image, MODEL)
print("done")