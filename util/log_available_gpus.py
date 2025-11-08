try:
    import torch
except Exception as _e:
    # Keep the import failure available but allow the module to be imported without crashing
    torch = None
    _TORCH_IMPORT_ERROR = _e


def _format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable GB string with raw bytes."""
    try:
        gb = float(bytes_val) / (1024 ** 3)
        return f"{gb:.2f} GB ({bytes_val} bytes)"
    except Exception:
        return f"{bytes_val} bytes"


def log_available_gpus():
    """Log available GPU information including total memory per device.

    Safe if PyTorch is not installed or CUDA is not available.
    """
    if torch is None:
        print("PyTorch is not available:", _TORCH_IMPORT_ERROR)
        return

    print('CUDA available:', torch.cuda.is_available())

    try:
        num_gpus = torch.cuda.device_count()
    except Exception as e:
        print('Error getting device count:', e)
        num_gpus = 0

    print('Number of CUDA devices:', num_gpus)

    if num_gpus > 0:
        for i in range(num_gpus):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception as e:
                name = f'Unknown ({e})'
            print(f'device name [{i}]:', name)

            # Try to get device properties (includes total_memory)
            try:
                props = torch.cuda.get_device_properties(i)
                total_mem = getattr(props, 'total_memory', None)
                if total_mem is not None:
                    print(f'  total memory: {_format_bytes(total_mem)}')
                else:
                    print('  total memory: Unknown')
            except Exception as e:
                print('  error reading properties:', e)

            # Report current process memory usage on that device (may be 0 if unused)
            try:
                # Use context manager to set device for reading memory stats
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                print(f'  memory allocated: {_format_bytes(allocated)}')
                print(f'  memory reserved:  {_format_bytes(reserved)}')
            except Exception as e:
                print('  error reading memory usage:', e)

    # Versions
    print('CUDA version:', getattr(torch.version, 'cuda', None))
    print('torch.version.hip:', getattr(torch.version, 'hip', None))

