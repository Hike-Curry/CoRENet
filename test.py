import argparse
import datetime
import random
import time
import numpy as np
import torch
from args import parser as training_parser
from checkpoint import read
from lib.ccsnet import build_model
from engine import make_loader, run_epoch, report


def main(cli):
    state = read(cli.checkpoint)
    config = state.get('config', {})
    if config.get('dataset') != cli.dataset or config.get('method') != 'ccsnet':
        raise ValueError('Checkpoint dataset or model does not match')
    for key in ('no_early_language', 'no_role_embedding', 'no_detail_refine', 'matched_upsampling'):
        if config.get(key, False):
            raise ValueError('Checkpoint is not the supported full model')
    if config.get('score_mode', 'region') != 'region' or not config.get('refine_conditioning', True):
        raise ValueError('Checkpoint is not the supported full model')
    if config.get('context_grid', 16) != 16 or config.get('mixture_temperature', 1.) != 1.:
        raise ValueError('Checkpoint uses incompatible decoder settings')
    argv = ['--dataset', cli.dataset]
    args = training_parser(argv).parse_args(argv)
    keys = ('swin_type', 'window_size', 'adapter_dim', 'num_candidates', 'max_tokens',
            'img_size', 'split_by', 'eval_sentences')
    if any(k not in config for k in keys):
        raise ValueError('Checkpoint is missing model or evaluation configuration')
    for key in keys:
        setattr(args, key, config[key])
    args.data_root = cli.data_root or args.data_root
    args.bert_dir, args.device, args.workers = cli.bert_dir, cli.device, cli.workers
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, _ = build_model(args, initialize=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to(args.device)
    loader = make_loader(args, 'test')
    started = time.perf_counter()
    metrics, _ = run_epoch(model, loader, args, 'test')
    report('Test', metrics)
    elapsed = time.perf_counter() - started
    print('Total test time', datetime.timedelta(seconds=int(elapsed)))
    print('Test time for one image %.4f' % (elapsed / metrics['samples']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CoRE-Net test')
    p.add_argument('--dataset', choices=['rrsisd', 'refsegrs'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root')
    p.add_argument('--bert-dir', default='bert_base_uncased')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
