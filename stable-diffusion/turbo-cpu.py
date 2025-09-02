from diffusers import AutoPipelineForText2Image
import torch
import re
import datetime

print("load model...")
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
pipe.to("cpu")

width = 512
height = 512


prompt = "old man with mustache, gray hair, white skin, only face"
safe_prompt = re.sub(r"[^a-zA-Z]", "_", prompt)

generationCount = 10
for i in range(generationCount):
    print(i + 1, "of " + str(generationCount) + ": Generating image for prompt:", prompt)
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=height, width=width).images[0]
    print("save paint...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(timestamp + "_" + safe_prompt + "_" + str(width) + "x" + str(height) + ".png")
    print("done")