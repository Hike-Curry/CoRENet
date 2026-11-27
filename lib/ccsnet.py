import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from loss.segmentation import per_mask_loss, candidate_quality, quality_loss
ROLES = ('object', 'modifier', 'anchor', 'anchor_modifier', 'relation', 'position')

def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU())

def grid(h, w, device, dtype):
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    return torch.stack((x, y), -1).reshape(1, h * w, 2).to(dtype)

class DenseLanguageFusion(nn.Module):

    def __init__(self, channels, dim):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, dim))
        self.geometry = nn.Linear(2, dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.output = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, channels))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, h, w, words, valid):
        visual = self.visual(x)
        query = visual + self.geometry(grid(h, w, x.device, visual.dtype))
        language = self.attention(query, words, words, key_padding_mask=~valid, need_weights=False)[0]
        return x + self.gate.sigmoid() * self.output(torch.cat((visual * language, language), -1))

class GroundedStage(nn.Module):

    def __init__(self, channels, depth, heads, window, drop_rates, dim, downsample):
        super().__init__()
        from lib.backbone import SwinTransformerBlock, PatchMerging
        self.window_size, self.shift_size = (window, window // 2)
        self.blocks = nn.ModuleList((SwinTransformerBlock(channels, heads, window_size=window, shift_size=0 if i % 2 == 0 else window // 2, drop_path=drop_rates[i]) for i in range(depth)))
        self.fusion = DenseLanguageFusion(channels, dim)
        self.downsample = PatchMerging(channels) if downsample else None

    def forward(self, x, h, w, words, valid, recompute):
        from lib.visual import visual_blocks
        run = lambda value: visual_blocks(self, value, h, w)
        x = checkpoint(run, x, use_reentrant=False) if recompute and self.training else run(x)
        run_fusion = lambda value, text: self.fusion(value, h, w, text, valid)
        x = checkpoint(run_fusion, x, words, use_reentrant=False) if recompute and self.training else run_fusion(x, words)
        following = self.downsample(x, h, w) if self.downsample is not None else x
        return (x, following)

class GroundedVision(nn.Module):

    def __init__(self, width=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window=7, dim=64, recompute=True, drop_path=0.2):
        super().__init__()
        from lib.backbone import PatchEmbed
        self.patch_embed = PatchEmbed(embed_dim=width, norm_layer=nn.LayerNorm)
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.layers = nn.ModuleList((GroundedStage(width * 2 ** i, depth, heads[i], window, rates[sum(depths[:i]):sum(depths[:i + 1])], dim, i < 3) for i, depth in enumerate(depths)))
        for i in range(4):
            self.add_module(f'norm{i}', nn.LayerNorm(width * 2 ** i))
        self.recompute = recompute

    def forward(self, image, words, valid):
        x = self.patch_embed(image)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        features = []
        for i, layer in enumerate(self.layers):
            out, x = layer(x, h, w, words, valid, self.recompute)
            features.append(getattr(self, f'norm{i}')(out).transpose(1, 2).reshape(len(x), -1, h, w))
            if layer.downsample is not None:
                h, w = ((h + 1) // 2, (w + 1) // 2)
        return features

class EvidenceDecoder(nn.Module):

    def __init__(self, width=128, dim=64, candidates=4):
        super().__init__()
        self.laterals = nn.ModuleList((nn.Conv2d(width * 2 ** i, dim, 1) for i in range(4)))
        self.smooth = nn.ModuleList((conv(dim, dim) for _ in range(4)))
        self.auxiliary = nn.ModuleList((nn.Conv2d(dim, 1, 1) for _ in range(4)))
        self.slots = nn.Parameter(torch.randn(candidates, dim) * 0.2)
        self.text_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.visual_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.geometry = nn.Linear(2, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.mask_query = nn.Linear(dim, dim)
        self.mask_bias = nn.Linear(dim, 1)
        self.score = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.rgb_detail = conv(3, dim // 2 if dim >= 16 else 8, stride=2)
        self.refine = conv(dim + (dim // 2 if dim >= 16 else 8) + 1, dim)
        self.boundary = nn.Conv2d(dim, 1, 1)
        self.residual = nn.Conv2d(dim, 1, 1)
        self.refine_gate = nn.Conv2d(dim, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features, words, valid, image):
        levels = [layer(x) for layer, x in zip(self.laterals, features)]
        pyramid = [None] * 4
        pyramid[3] = self.smooth[3](levels[3])
        for i in (2, 1, 0):
            pyramid[i] = self.smooth[i](levels[i] + F.interpolate(pyramid[i + 1], levels[i].shape[-2:], mode='bilinear', align_corners=False))
        x = pyramid[0]
        full = (words * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        queries = self.slots[None] + full[:, None]
        queries = queries + self.text_attention(queries, words, words, key_padding_mask=~valid, need_weights=False)[0]
        h, w = (min(16, x.shape[-2]), min(16, x.shape[-1]))
        context = F.adaptive_avg_pool2d(x, (h, w)).flatten(2).transpose(1, 2)
        context = context + self.geometry(grid(h, w, x.device, x.dtype))
        queries = self.query_norm(queries + self.visual_attention(queries, context, context, need_weights=False)[0])
        masks = torch.einsum('bkd,bdhw->bkhw', self.mask_query(queries), x) / math.sqrt(x.shape[1])
        masks = masks + self.mask_bias(queries)[..., None]
        probability = masks.float().sigmoid()
        regions = torch.einsum('bkhw,bdhw->bkd', probability, x.float()) / probability.sum((-2, -1))[..., None].clamp_min(1e-06)
        scores = self.score(torch.cat((queries, regions.to(queries.dtype), full[:, None].expand_as(queries)), -1))[..., 0]
        weights = scores.float().softmax(-1)
        mixture = (probability * weights[..., None, None]).sum(1, keepdim=True)
        coarse = torch.logit(mixture.clamp(1e-05, 1 - 1e-05))
        detail = self.rgb_detail(image)
        high = F.interpolate(x, detail.shape[-2:], mode='bilinear', align_corners=False)
        prior = F.interpolate(coarse, detail.shape[-2:], mode='bilinear', align_corners=False)
        semantic = high
        condition = prior.to(high.dtype)
        refined = self.refine(torch.cat((semantic, detail, condition), 1))
        boundary = self.boundary(refined).float()
        foreground = prior + self.refine_gate(refined).float().sigmoid() * self.residual(refined).float()
        size = image.shape[-2:]
        resize = lambda value: F.interpolate(value.float(), size, mode='bilinear', align_corners=False)
        foreground = resize(foreground)
        return dict(logits=torch.cat((torch.zeros_like(foreground), foreground), 1), candidate_logits=resize(masks), candidate_weights=weights, candidate_scores=scores, pre_refine_logit=resize(coarse)[:, 0], auxiliary_logits=[head(level).float() for head, level in zip(self.auxiliary, pyramid)], boundary_logits=resize(boundary)[:, 0])

class CCSNet(nn.Module):

    def __init__(self, text_encoder, width=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window=7, dim=64, candidates=4, language_dim=768, recompute=True, drop_path=0.2):
        super().__init__()
        self.text_encoder = text_encoder
        self.words = nn.Linear(language_dim, dim)
        self.role_embedding = nn.Parameter(torch.randn(len(ROLES), dim) * 0.02)
        self.visual = GroundedVision(width, depths, heads, window, dim, recompute, drop_path)
        self.decoder = EvidenceDecoder(width, dim, candidates)

    def forward(self, batch):
        valid = batch['attention'].bool()
        language = self.text_encoder(batch['ids'], attention_mask=valid)[0]
        words = self.words(language)
        roles = torch.stack([batch['roles']['comp_' + key] for key in ROLES], -1).to(words.dtype)
        words = words + roles @ self.role_embedding
        features = self.visual(batch['image'], words, valid)
        return self.decoder(features, words, valid, batch['image'])

    def optimizer_groups(self, args):
        return [dict(params=self.visual.parameters(), lr=args.base_lr), dict(params=self.text_encoder.parameters(), lr=args.bert_lr), dict(params=list(self.words.parameters()) + [self.role_embedding] + list(self.decoder.parameters()), lr=args.new_lr)]

def loss_function(result, truth, args):
    truth = truth.float()
    seg = per_mask_loss(result['logits'][:, 1], truth).mean()
    candidates = result['candidate_logits']
    candidate = per_mask_loss(candidates, truth[:, None].expand_as(candidates)).min(1).values.mean()
    quality, _ = quality_loss(result['candidate_scores'], candidate_quality(candidates, truth), args.quality_temperature, args.quality_minimum)
    mixture = per_mask_loss(result['pre_refine_logit'], truth).mean()
    union = torch.logit(candidates.float().sigmoid().amax(1).clamp(1e-05, 1 - 1e-05))
    union_loss = per_mask_loss(union, truth).mean()
    auxiliary = sum((per_mask_loss(F.interpolate(value, truth.shape[-2:], mode='bilinear', align_corners=False)[:, 0], truth).mean() for value in result['auxiliary_logits'])) / len(result['auxiliary_logits'])
    target = truth[:, None]
    edges = (F.max_pool2d(target, 3, 1, 1) - -F.max_pool2d(-target, 3, 1, 1))[:, 0]
    boundary = per_mask_loss(result['boundary_logits'], edges).mean()
    regional = candidate + quality + auxiliary
    structure = mixture + union_loss + boundary
    total = (10 * seg + 2 * regional + structure) / 13
    return (total, dict(seg=seg, candidate=candidate, quality=quality, mixture=mixture, multiscale=auxiliary, union=union_loss, boundary=boundary))

def initialize_visual(model, path):
    from checkpoint import read
    raw = read(path)
    state = raw.get('model', raw.get('state_dict', raw))
    state = {key.removeprefix('module.'): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if 'patch_embed.proj.weight' not in state or any((key.startswith(('backbone.', 'base.', 'visual.', 'classifier.')) for key in state)):
        raise ValueError('--vision-init requires generic Swin classification weights, not a task-specific segmentation checkpoint')
    target = model.visual.state_dict()
    for key, value in list(state.items()):
        if key in target and key.endswith('relative_position_bias_table') and (value.shape != target[key].shape):
            old_size, new_size = (math.isqrt(value.shape[0]), math.isqrt(target[key].shape[0]))
            if old_size ** 2 == value.shape[0] and new_size ** 2 == target[key].shape[0] and (value.shape[1] == target[key].shape[1]):
                table = value.T.reshape(1, value.shape[1], old_size, old_size).float()
                state[key] = F.interpolate(table, (new_size, new_size), mode='bicubic', align_corners=False).reshape(value.shape[1], -1).T.to(value.dtype)
    required = {key: value for key, value in target.items() if '.fusion.' not in key and (not key.startswith(('norm0.', 'norm1.', 'norm2.', 'norm3.'))) and (not key.endswith('relative_position_index'))}
    missing = [key for key in required if key not in state]
    mismatch = [key for key in required if key in state and state[key].shape != required[key].shape]
    if missing or mismatch:
        raise ValueError(f'Generic Swin architecture mismatch; missing={missing[:5]}, shape={mismatch[:5]}. Match --window-size and --swin-type; renaming cannot fix it.')
    matched = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape and (not key.endswith('relative_position_index'))}
    for suffix in ('weight', 'bias'):
        key = 'norm.' + suffix
        if key in state and state[key].shape == target['norm3.' + suffix].shape:
            matched['norm3.' + suffix] = state[key]
    model.visual.load_state_dict(matched, strict=False)
    return dict(source=str(path), kind='generic_classification_swin', loaded_tensors=len(matched), task_checkpoint_used=False, new_fusion_and_decoder='random')

def build_model(args, initialize=True):
    from transformers import BertConfig, BertModel
    spec = {'tiny': (96, (2, 2, 6, 2), (3, 6, 12, 24)), 'small': (96, (2, 2, 18, 2), (3, 6, 12, 24)), 'base': (128, (2, 2, 18, 2), (4, 8, 16, 32)), 'large': (192, (2, 2, 18, 2), (6, 12, 24, 48))}
    width, depths, heads = spec[args.swin_type]
    config = BertConfig.from_pretrained(args.bert_dir, local_files_only=True)
    text = BertModel.from_pretrained(args.bert_dir, local_files_only=True) if initialize and (not args.random_init) else BertModel(config)
    if args.gradient_checkpointing:
        text.gradient_checkpointing_enable()
    model = CCSNet(text, width, depths, heads, args.window_size, args.adapter_dim, args.num_candidates, config.hidden_size, args.gradient_checkpointing)
    report = dict(task_checkpoint_used=False, epoch_minus_one=False, bert='generic_bert' if initialize and (not args.random_init) else 'random_or_resume')
    if initialize and (not args.random_init):
        report['visual'] = initialize_visual(model, args.vision_init)
    else:
        report['visual'] = 'random_or_resume'
    return (model, report)
