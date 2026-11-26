import json
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image

class Metrics:

    def __init__(self):
        self.n = self.i = self.u = self.sum_iou = self.loss = self.zero = 0
        self.precision = {t: 0 for t in (0.5, 0.6, 0.7, 0.8, 0.9)}

    def update(self, logits, target, loss):
        pred = logits.argmax(1)
        intersection = ((pred == 1) & (target == 1)).sum((1, 2)).double()
        union = ((pred == 1) | (target == 1)).sum((1, 2)).double()
        iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union))
        self.n += len(pred)
        self.i += intersection.sum().item()
        self.u += union.sum().item()
        self.sum_iou += iou.sum().item()
        self.zero += int((iou == 0).sum())
        for threshold in self.precision:
            self.precision[threshold] += int((iou >= threshold).sum())
        self.loss += float(loss) * len(pred)
        return (pred, iou)

    def result(self, partial):
        return dict(loss=self.loss / max(self.n, 1), mIoU=100 * self.sum_iou / max(self.n, 1), oIoU=100 * self.i / max(self.u, 1), zero_iou=self.zero, samples=self.n, partial=partial, **{f'Pr@{t:.1f}': 100 * count / max(self.n, 1) for t, count in self.precision.items()})

def atomic_save(path, payload):
    path = Path(path)
    size = sum((v.numel() * v.element_size() for v in payload['model'].values()))
    required = size * (4 if 'optimizer' in payload else 1) + 128 * 1024 ** 2
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(f'Insufficient free disk for atomic checkpoint; need about {required / 2 ** 30:.2f} GiB FREE. Existing best is untouched.')
    tmp = path.with_suffix('.tmp')
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

def export_zero(batch, pred, index, directory, slot, epoch, boxes=None):
    directory.mkdir(parents=True, exist_ok=True)
    image = batch['image'][index].detach().cpu() * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    rgb = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
    gt = batch['mask'][index].detach().cpu().numpy().astype('uint8') * 255
    mask = pred[index].detach().cpu().numpy().astype('uint8') * 255
    panels = np.concatenate([rgb, np.repeat(gt[..., None], 3, -1), np.repeat(mask[..., None], 3, -1)], axis=1)
    stem = directory / f'case_{slot:03d}'
    Image.fromarray(panels).save(str(stem) + '.png')
    info = dict(epoch=epoch, ref_id=batch['ref_id'][index], text=batch['text'][index], image_path=batch['image_path'][index], panels='RGB | GT | prediction', box=boxes[index].detach().cpu().tolist() if boxes is not None else None)
    Path(str(stem) + '.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')

def worker_seed(worker_id):
    value = torch.initial_seed() % 2 ** 32
    random.seed(value)
    np.random.seed(value)
