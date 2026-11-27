import argparse


def parser(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--dataset', choices=['rrsisd', 'refsegrs'], required=True)
    selected, _ = selector.parse_known_args(argv)
    ref = selected.dataset == 'refsegrs'
    p = argparse.ArgumentParser(description='CoRE-Net')
    p.add_argument('--dataset', choices=['rrsisd', 'refsegrs'], required=True)
    p.add_argument('--data-root', default='data/' + selected.dataset)
    p.add_argument('--bert-dir', default='bert_base_uncased')
    p.add_argument('--vision-init', default='pretrained_weights/swin_base_patch4.pth')
    p.add_argument('--output', default='outputs/' + selected.dataset)
    p.add_argument('--epochs', type=int, default=150 if ref else 100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=2401)
    p.add_argument('--img-size', type=int, default=480)
    p.add_argument('--swin-type', choices=['tiny', 'small', 'base', 'large'], default='base')
    p.add_argument('--window-size', type=int, choices=[7, 12], default=12)
    p.add_argument('--base-lr', type=float, default=3e-5 if ref else 2e-6)
    p.add_argument('--bert-lr', type=float, default=1e-5 if ref else 1e-6)
    p.add_argument('--new-lr', type=float, default=1e-4 if ref else 1e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--random-init', action='store_true')
    p.set_defaults(method='ccsnet', max_tokens=20 if ref else 22, split_by='unc',
                   eval_sentences='first', adapter_dim=64, num_candidates=4,
                   quality_temperature=.15, quality_minimum=.1,
                   select_best_by='val')
    return p
