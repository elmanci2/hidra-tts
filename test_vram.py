import torch

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {total:.2f} GB")
else:
    print("NO CUDA")
