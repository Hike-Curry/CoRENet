import torch
from torch.nn import functional as F

def per_mask_loss(logits, target):
    target = target.float()
    probability = logits.float().sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none').mean((-2, -1))
    dice = 1 - (2 * (probability * target).sum((-2, -1)) + 1) / (probability.sum((-2, -1)) + target.sum((-2, -1)) + 1)
    return ce + dice

@torch.no_grad()
def candidate_quality(logits, truth):
    pred = logits.detach() > 0
    target = truth[:, None].bool()
    intersection = (pred & target).sum((-2, -1)).float()
    union = (pred | target).sum((-2, -1)).clamp_min(1)
    return intersection / union

def quality_loss(scores, quality, temperature=0.15, minimum=0.1):
    quality = quality.detach()
    maximum = quality.max(1).values
    confidence = maximum * (maximum >= minimum)
    target = (quality / temperature).softmax(1)
    rank = -(target * scores.float().log_softmax(1)).sum(1)
    rank = (rank * confidence).mean()
    calibration = F.binary_cross_entropy_with_logits(scores.float(), quality)
    return (rank + 0.25 * calibration, confidence)
