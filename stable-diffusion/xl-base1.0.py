from diffusers import DiffusionPipeline
import torch

modelName = "stabilityai/stable-diffusion-xl-base-1.0"

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

#for cpu
#device = torch.device('cpu')

if(device.type == 'cpu'):
    pipe = DiffusionPipeline.from_pretrained(modelName, use_safetensors=True)
else:
    pipe = DiffusionPipeline.from_pretrained(modelName, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.to(device)

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images[0].save("test.png")