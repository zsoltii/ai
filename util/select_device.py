import torch
import sys

def select_device():
    if '--force-cpu' in sys.argv:
        print('Forced CPU by command line argument')
        return torch.device('cpu')
    else:
        return select_device_automatic()


def select_device_automatic():
    # Default: auto-detect ROCm/AMD GPU, else CPU
    if getattr(torch.version, 'hip', None) is not None:
        print('ROCm detected, using device: cuda')
        return torch.device('cuda')  # ROCm uses 'cuda' as device string
    elif torch.version.cuda is not None:
        print(f'NVIDIA CUDA detected (version {torch.version.cuda}), using device: cuda')
        return torch.device('cuda')
    else:
        print('No ROCm or NVIDIA CUDA detected, using CPU')
        return torch.device('cpu')
