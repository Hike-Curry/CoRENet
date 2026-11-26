from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from bert.tokenization_bert import BertTokenizer
from data.text import ScopedRoles
from data.rrsisd import legacy_masks, MEAN, STD
CLASSES = {'road', 'vehicle', 'car', 'van', 'building', 'truck', 'trailer', 'bus', 'road marking', 'bikeway', 'sidewalk', 'tree', 'low vegetation', 'impervious surface'}

class RefSegRSDataset(Dataset):

    def __init__(self, root, bert_dir, split='train', size=480, max_tokens=20, split_by='unc', eval_sentences='first', legacy_occlusion=False):
        if split not in ('train', 'val', 'test') or max_tokens != 20 or legacy_occlusion:
            raise ValueError('RefSegRS requires train/val/test, max_tokens=20, no legacy occlusion')
        root = Path(root)
        if (root / 'refsegrs').is_dir():
            root /= 'refsegrs'
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
        fast = BertTokenizerFast.from_pretrained(bert_dir, local_files_only=True, clean_up_tokenization_spaces=False)
        roles = ScopedRoles(CLASSES)
        self.size, self.samples = (size, [])
        for line in (root / f'output_phrase_{split}.txt').read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            identity, text = line.split(' ', 1)
            image, mask = (root / 'images' / f'{identity}.tif', root / 'masks' / f'{identity}.tif')
            for path in (image, mask):
                if not path.is_file():
                    raise FileNotFoundError(path)
            encoded = tokenizer.encode(text, add_special_tokens=True)[:max_tokens]
            offsets = fast(text, return_offsets_mapping=True)
            if encoded != offsets['input_ids'][:max_tokens]:
                raise ValueError(f'Legacy/fast BERT token mismatch: {identity}')
            ids = torch.zeros(max_tokens, dtype=torch.long)
            attention = torch.zeros_like(ids)
            ids[:len(encoded)] = torch.tensor(encoded)
            attention[:len(encoded)] = 1
            target, position = legacy_masks(text, attention, CLASSES)
            offsets = offsets['offset_mapping'][:max_tokens]
            offsets += [(0, 0)] * (max_tokens - len(offsets))
            self.samples.append(dict(image_path=str(image.resolve()), mask_path=str(mask.resolve()), ref_id=str(len(self.samples)) + '_' + identity, text=text, ids=ids, attention=attention, target=target, position=position, roles=roles(text, offsets)))
        if not self.samples:
            raise ValueError(f'Empty split: {root} / {split}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index].copy()
        with Image.open(record['image_path']) as source:
            image = source.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = cv2.imread(record.pop('mask_path'), 2)
        if mask is None:
            raise ValueError(f"Unreadable mask: {self.samples[index]['mask_path']}")
        mask = Image.fromarray((mask > 50).astype('uint8')).resize((self.size, self.size), Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return dict(image=(tensor - MEAN) / STD, mask=torch.from_numpy(np.asarray(mask).copy()).long(), **record)
