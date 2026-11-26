import torch

def read(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
