import torch

def log_available_gpus():
    print(f'CUDA available:', torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print(f'Number of CUDA devices:', num_gpus)
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f'device name [{i}]:', torch.cuda.get_device_name(i))
    print(f'CUDA version:', torch.version.cuda)
    print('torch.version.hip:', getattr(torch.version, 'hip', None))
