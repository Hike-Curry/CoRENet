import math
import torch

def visual_blocks(layer, x, h, w):
    from lib.backbone import window_partition
    ws, ss = (layer.window_size, layer.shift_size)
    hp, wp = (math.ceil(h / ws) * ws, math.ceil(w / ws) * ws)
    image_mask = torch.zeros((1, hp, wp, 1), device=x.device)
    count = 0
    for a in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
        for b in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            image_mask[:, a, b] = count
            count += 1
    windows = window_partition(image_mask, ws).view(-1, ws * ws)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
    for block in layer.blocks:
        block.H, block.W = (h, w)
        x = block(x, mask)
    return x
