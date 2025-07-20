import torch
# Set PyTorch to use 6 CPU threads
torch.set_num_threads(6)
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

print("load model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

# Load images as PIL images
print("load image...")
image = Image.open("images/image.png").convert("RGB")
print("load mask...")
mask_image = Image.open("images/mask.png").convert("L")  # L mode for mask

print("generate paint for prompt: " + prompt)
# The mask structure is white for inpainting and black for keeping as is
result = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
print("save paint...")
result.save("./yellow_cat_on_park_bench.png")
print("done")