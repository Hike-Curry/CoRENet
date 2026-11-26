import json
import pickle
import random
import re
from pathlib import Path
import nltk
nltk.data.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nltk_data'))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from data.text import RoleParser, decode_segmentation
CLASSES = {'airplane', 'airport', 'golf field', 'expressway service area', 'baseball field', 'stadium', 'ground track field', 'storage tank', 'basketball court', 'chimney', 'tennis court', 'overpass', 'train station', 'ship', 'expressway toll station', 'dam', 'harbor', 'bridge', 'vehicle', 'windmill'}
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def legacy_masks(text, attention, categories=CLASSES):
    target, position = (torch.zeros_like(attention), torch.zeros_like(attention))
    words = nltk.word_tokenize(text)
    for cls in categories:
        if re.findall(cls, text):
            parts = nltk.word_tokenize(cls)
            start = next((i for i, word in enumerate(words) if re.findall(parts[0], word)), 0)
            target[start + 1:start + len(parts) + 1] = 1
    grammar = 'PP: {<IN><DT>?<JJ.*>?<NN>}\n    {<IN><DT>?<JJ.*>?<JJ>}\n    {<IN><DT>?<JJ.*><VBD>}'
    tree = nltk.RegexpParser(grammar).parse(nltk.pos_tag(words))
    for node in tree.subtrees():
        if node.label() != 'PP':
            continue
        phrase = ' '.join((word for word, _ in node.leaves()))
        if re.findall('of', phrase):
            continue
        parts = nltk.word_tokenize(phrase)
        start = next((i for i, word in enumerate(words) if parts[0] == word), 0)
        position[start + 1:start + len(parts) + 1] = 1
    if not position.any():
        position = attention.clone()
    return (target, position)

def make_roles(parser, text, offsets):
    roles = parser(text, offsets)
    words = set(re.findall('\\b\\w+\\b', text.lower()))
    nouns = list(parser.nouns.finditer(text.lower()))
    ordinal = bool(words & {'first', 'second', 'third', 'fourth', 'last', 'leftmost', 'rightmost'})
    safe = len(nouns) == 1 and (not parser.REL.search(text.lower())) and (not ordinal)
    safe = safe and (not bool(words & {'between', 'among'})) and bool(roles['target'].any())
    x = (-1 if words & {'left', 'west'} else 0) + (1 if words & {'right', 'east'} else 0)
    y = (-1 if words & {'top', 'upper', 'north'} else 0) + (1 if words & {'bottom', 'lower', 'south'} else 0)
    roles['image_prior'] = torch.tensor([x, y], dtype=torch.float32) if safe else torch.zeros(2)
    roles['force_global'] = torch.tensor(ordinal or bool(words & {'between'}))
    return roles

class ZoomDataset(Dataset):

    def __init__(self, root, bert_dir, split='train', size=480, max_tokens=22, split_by='unc', eval_sentences='first', legacy_occlusion=False):
        self.root = Path(root)
        if (self.root / 'rrsisd').is_dir():
            self.root = self.root / 'rrsisd'
        with (self.root / f'refs({split_by}).p').open('rb') as handle:
            refs = pickle.load(handle)
        instances = json.loads((self.root / 'instances.json').read_text(encoding='utf8'))
        self.images = {x['id']: x for x in instances['images']}
        self.annotations = {x['id']: x for x in instances['annotations']}
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir, local_files_only=True, clean_up_tokenization_spaces=False)
        parser = RoleParser([x['name'] for x in instances['categories']])
        self.size, self.training = (size, split == 'train')
        chosen = [r for r in refs if r['split'] == split]
        self.occluded = set(random.sample([r['ref_id'] for r in chosen], int(0.2 * len(chosen)))) if self.training and legacy_occlusion else set()
        self.samples = []
        cache = {}
        for ref in chosen:
            expressions = []
            for sent in ref['sentences']:
                text = sent['raw']
                if text not in cache:
                    enc = tokenizer(text, return_offsets_mapping=True, truncation=False)
                    length = min(len(enc['input_ids']), max_tokens)
                    ids = torch.zeros(max_tokens, dtype=torch.long)
                    att = torch.zeros_like(ids)
                    ids[:length] = torch.tensor(enc['input_ids'][:length])
                    att[:length] = 1
                    offsets = enc['offset_mapping'][:length] + [(0, 0)] * (max_tokens - length)
                    target, position = legacy_masks(text, att)
                    cache[text] = dict(ids=ids, attention=att, target=target, position=position, roles=make_roles(parser, text, offsets), text=text)
                expressions.append(cache[text])
            if not expressions:
                raise ValueError(f"No expressions for reference {ref['ref_id']}")
            if not self.training and eval_sentences == 'all':
                self.samples.extend(((ref, [e]) for e in expressions))
            else:
                self.samples.append((ref, expressions if self.training else expressions[:1]))
        if not self.samples:
            raise ValueError(f'Empty split {split}: {self.root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ref, expressions = self.samples[index]
        name = Path(self.images[ref['image_id']]['file_name']).name
        candidates = [self.root / 'images' / name, self.root / 'JPEGImages' / name, self.root.parent / 'images' / 'rrsisd' / 'JPEGImages' / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Image not found; tried: {candidates}')
        with Image.open(path) as source:
            image = np.asarray(source.convert('RGB')).copy()
        h, w = image.shape[:2]
        mask = decode_segmentation(self.annotations[ref['ann_id']]['segmentation'], h, w)
        if ref['ref_id'] in self.occluded:
            for _ in range(random.randint(20, 60)):
                x, y = (random.randint(0, max(w - 32, 0)), random.randint(0, max(h - 32, 0)))
                image[y:y + 32, x:x + 32] = 0
        global_image = Image.fromarray(image).resize((self.size, self.size), Image.Resampling.BILINEAR)
        global_image = torch.from_numpy(np.asarray(global_image).copy()).permute(2, 0, 1).float() / 255
        global_mask = Image.fromarray(mask.astype('uint8')).resize((self.size, self.size), Image.Resampling.NEAREST)
        expression = random.choice(expressions) if self.training else expressions[0]
        return dict(image=(global_image - MEAN) / STD, native=torch.from_numpy(image).permute(2, 0, 1), native_mask=torch.from_numpy(mask.astype('uint8').copy())[None], mask=torch.from_numpy(np.asarray(global_mask).copy()).long(), ref_id=str(ref['ref_id']), image_path=str(path.resolve()), **expression)

def collate(samples):
    result = {}
    for key in samples[0]:
        if key in ('native', 'native_mask', 'text', 'image_path', 'ref_id', 'category_text'):
            result[key] = [s[key] for s in samples]
        elif key == 'roles':
            result[key] = {r: torch.stack([s[key][r] for s in samples]) for r in samples[0][key]}
        else:
            result[key] = torch.stack([s[key] for s in samples])
    return result

def move_batch(batch, device):
    return {k: {r: v.to(device) for r, v in value.items()} if isinstance(value, dict) else value.to(device) if isinstance(value, torch.Tensor) else value for k, value in batch.items()}
