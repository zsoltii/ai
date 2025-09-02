import datetime
import re

import torch
from diffusers import AutoPipelineForText2Image

# Detect ROCm/AMD GPU and set device accordingly
if torch.version.hip is not None:
    device = torch.device('cuda')  # ROCm uses 'cuda' as device string
    print('ROCm detected, using device:', device)
else:
    device = torch.device('cpu')
    print('No ROCm detected, using CPU')

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'device name [0]:', torch.cuda.get_device_name(0))
print(torch.version.cuda)
print('torch.version.hip:', getattr(torch.version, 'hip', None))

print("load model...")
model = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
print(f"Model loaded: {getattr(model, 'name_or_path', str(model))}")
model.enable_model_cpu_offload()
model.enable_sequential_cpu_offload()
model.to(device)

prompt = "young children playing on playground, cinematic lights"
safe_prompt = re.sub(r"[^a-zA-Z]", "_", prompt)

generationCount = 3
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = model(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    print("save paint...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image.save(timestamp + "_" + safe_prompt + ".png")
    print(i + 1, "of " + str(generationCount) + ": Generating image done for prompt:", prompt)

print("done")
