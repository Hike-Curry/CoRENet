import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import GroundedRef, GroundedRRS
from data.rrsisd import collate, move_batch
from lib.ccsnet import build_model, loss_function
from utils import Metrics, worker_seed
from artifacts import preserve_rng, save_best


def make_loader(args, split):
    factory = GroundedRef if args.dataset == 'refsegrs' else GroundedRRS
    dataset = factory(root=args.data_root, bert_dir=args.bert_dir, split=split,
                      size=args.img_size, max_tokens=args.max_tokens,
                      split_by=args.split_by, eval_sentences=args.eval_sentences,
                      legacy_occlusion=False)
    return DataLoader(dataset, batch_size=args.batch_size if split == 'train' else 1,
                      shuffle=split == 'train', num_workers=args.workers,
                      collate_fn=collate, worker_init_fn=worker_seed,
                      pin_memory=args.device.startswith('cuda'))


def report(split, metrics):
    print(f'{split}: Mean IoU is {metrics["mIoU"]:.4f}')
    for threshold in (.5, .6, .7, .8, .9):
        print('    precision@%.1f = %.4f' % (threshold, metrics[f'Pr@{threshold:.1f}']))
    print('    overall IoU = %.4f' % metrics['oIoU'], flush=True)


def run_epoch(model, loader, args, split, epoch=0, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metric, zeros, offset = Metrics(), [], 0
    device = torch.device(args.device)
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device.type, enabled=training and args.amp and device.type == 'cuda'):
                result = model(batch)
                loss, _ = loss_function(result, batch['mask'], args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {split}, step {step}')
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not scaler.is_enabled() and not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient')
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous:
                    scheduler.step()
        _, iou = metric.update(result['logits'].detach(), batch['mask'], loss.detach())
        if not training:
            zeros.extend(offset + i for i in (iou == 0).nonzero().flatten().tolist())
        offset += len(iou)
        if step % (50 if training else 100) == 0:
            header = f'Epoch: [{epoch}]' if training else split.capitalize()
            print(f'{header} [{step}/{len(loader)}] loss: {loss.item():.4f}', flush=True)
    if metric.n == 0:
        raise ValueError(f'Empty {split} dataset')
    metrics = metric.result(False)
    metrics['elapsed_seconds'] = time.perf_counter() - started
    return metrics, zeros


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid epochs, batch size or workers')
    if min(args.base_lr, args.bert_lr, args.new_lr) <= 0 or args.weight_decay < 0 or args.img_size < 32:
        raise ValueError('Invalid learning rate, weight decay or image size')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or a GPU instance')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new --output directory to preserve existing results')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, initialization = build_model(args, initialize=True)
    model.to(args.device)
    train_loader, val_loader = [make_loader(args, split) for split in ('train', 'val')]
    optimizer = torch.optim.AdamW(model.optimizer_groups(args), weight_decay=args.weight_decay)
    steps = len(train_loader)
    def multiplier(step):
        progress = min(1., step / max(1, steps * args.epochs))
        return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and args.device.startswith('cuda'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf8')
    (output / 'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf8')
    best = -1.
    print(f'Image size: {args.img_size}\nTraining samples: {len(train_loader.dataset)}', flush=True)
    for epoch in range(args.epochs):
        train_metrics, _ = run_epoch(model, train_loader, args, 'train', epoch, optimizer, scheduler, scaler)
        with preserve_rng():
            val_metrics, _ = run_epoch(model, val_loader, args, 'val')
        record = dict(epoch=epoch, train=train_metrics, val=val_metrics, selected_by='val.mIoU')
        for split in ('train', 'val'):
            report(split.upper(), record[split])
        if val_metrics['mIoU'] > best:
            best = val_metrics['mIoU']
            save_best(output, model, optimizer, scheduler, scaler, args, epoch, best, steps,
                      record)
            print(f'Best validation mIoU: {best:.4f}', flush=True)
        record['best_mIoU'] = best
        with (output / 'metrics.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(record) + '\n')
