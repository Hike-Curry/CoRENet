import re
import torch
from pycocotools import mask as mask_api

def decode_segmentation(seg, height, width):
    from pycocotools import mask as mask_api
    if isinstance(seg, list) and seg and isinstance(seg[0], dict):
        rle = [mask_api.frPyObjects(item, height, width) if isinstance(item['counts'], list) else item for item in seg]
    elif isinstance(seg, list):
        rle = mask_api.frPyObjects(seg, height, width)
    elif isinstance(seg['counts'], list):
        rle = mask_api.frPyObjects(seg, height, width)
    else:
        rle = seg
    mask = mask_api.decode(rle)
    return mask.any(-1) if mask.ndim == 3 else mask > 0

class RoleParser:
    ATTR = set('white black red blue gray grey green yellow brown dark bright light small smaller smallest large larger largest big tiny long short wide narrow round circular rectangular square linear curved oval first second third single double'.split())
    DIRECTIONS = {'left': 1, 'west': 1, 'right': 2, 'east': 2, 'above': 3, 'top': 3, 'upper': 3, 'north': 3, 'below': 4, 'bottom': 4, 'lower': 4, 'south': 4}
    REL = re.compile('\\b(?:(?:to\\s+)?(?:the\\s+)?(?:left|right|north|south|east|west)\\s+of|above|below|near|beside|next\\s+to|adjacent\\s+to)\\b')

    def __init__(self, categories=()):
        defaults = ['airplane', 'airport', 'golf field', 'expressway service area', 'baseball field', 'stadium', 'ground track field', 'storage tank', 'basketball court', 'chimney', 'tennis court', 'overpass', 'train station', 'ship', 'expressway toll station', 'dam', 'harbor', 'bridge', 'vehicle', 'windmill', 'building', 'road', 'river', 'tree', 'car', 'plane', 'tank', 'house', 'runway', 'pool']
        nouns = sorted(set(defaults + [x.lower() for x in categories]), key=len, reverse=True)
        self.nouns = re.compile('\\b(?:' + '|'.join((re.escape(x) for x in nouns)) + ')(?:s|es)?\\b')

    def __call__(self, text, offsets):
        text = text.lower()
        roles = {k: torch.zeros(len(offsets)) for k in ('target', 'target_attr', 'anchor', 'anchor_attr')}
        roles['spatial'] = torch.zeros(len(offsets))
        roles.update(direction=torch.tensor(0), absolute=torch.tensor(0))
        spatial_words = set(self.DIRECTIONS) | set('near beside between inside outside around adjacent along middle center central next'.split())
        for word in re.finditer('\\b\\w+\\b', text):
            if word.group() in spatial_words:
                for i, (a, b) in enumerate(offsets):
                    if b > a and a < word.end() and (b > word.start()):
                        roles['spatial'][i] = 1
        nouns = list(self.nouns.finditer(text))
        if not nouns:
            return roles

        def mark(key, start, end):
            for i, (a, b) in enumerate(offsets):
                if b > a and a < end and (b > start):
                    roles[key][i] = 1
        relations = list(self.REL.finditer(text))
        target = nouns[0]
        if relations and relations[0].start() < target.start() or re.search('\\bbetween\\b', text):
            return roles
        mark('target', *target.span())
        relation = relations[0] if len(relations) == 1 else None
        anchor = next((n for n in nouns[1:] if relation and n.start() >= relation.end()), None)
        if anchor and len(nouns) == 2:
            mark('anchor', *anchor.span())
            direction = next((self.DIRECTIONS[w] for w in relation.group().split() if w in self.DIRECTIONS), 0)
            roles['direction'] = torch.tensor(direction)
        else:
            anchor = None
        for word in re.finditer('\\b\\w+\\b', text):
            if word.group() in self.ATTR:
                if relation and word.start() >= relation.end():
                    if anchor:
                        mark('anchor_attr', *word.span())
                elif len(nouns) == 1 or relation:
                    mark('target_attr', *word.span())
        if not relations and len(nouns) == 1:
            directions = [self.DIRECTIONS[w.group()] for w in re.finditer('\\b\\w+\\b', text) if w.group() in self.DIRECTIONS]
            if len(set(directions)) == 1:
                roles['absolute'] = torch.tensor(directions[0])
        if not roles['anchor'].any():
            roles['direction'].zero_()
        if not roles['target'].any():
            for value in roles.values():
                value.zero_()
        return roles

class ScopedRoles:
    RELATION = re.compile('\\b(?:(?:(?:upper|lower|top|bottom)\\s+)?(?:left|right)\\s+of|(?:north|south|east|west)\\s+of|above|below|near|beside|next\\s+to|adjacent\\s+to|(?:bigger|larger|smaller|taller|shorter)\\s+than|similar\\s+in\\s+size\\s+to)\\b')
    POS = set('left right top bottom upper lower north south east west middle center central above below'.split())
    STOP = set('a an the is are was were be being of on at in to from and or with that which it its located situated corner side part very slightly little than similar size'.split())
    MASKS = ('entity', 'attribute', 'anchor_entity', 'anchor_attribute', 'relation', 'anchor_position', 'target_position')

    def __init__(self, categories=()):
        self.nouns = RoleParser(categories).nouns

    def __call__(self, text, offsets):
        text = text.lower()
        roles = {'rel_' + key: torch.zeros(len(offsets)) for key in self.MASKS}
        roles.update(rel_vector=torch.zeros(2), rel_size=torch.tensor(0.0), rel_anchor_xy=torch.zeros(2), rel_target_xy=torch.zeros(2), rel_valid=torch.tensor(False))
        nouns = list(self.nouns.finditer(text))
        if not nouns:
            return roles

        def mark(key, start, end):
            for i, (a, b) in enumerate(offsets):
                if b > a and a < end and (b > start):
                    roles['rel_' + key][i] = 1

        def vector(words):
            return torch.tensor([float(bool(words & {'right', 'east'})) - float(bool(words & {'left', 'west'})), float(bool(words & {'bottom', 'lower', 'south', 'below'})) - float(bool(words & {'top', 'upper', 'north', 'above'}))])

        def clause(start, end, attribute, position):
            words = set()
            for token in re.finditer('\\b[\\w-]+\\b', text[start:end]):
                a, b = (start + token.start(), start + token.end())
                word = token.group()
                if any((a < noun.end() and b > noun.start() for noun in nouns)):
                    continue
                if word in self.POS:
                    mark(position, a, b)
                    words.add(word)
                elif word not in self.STOP:
                    mark(attribute, a, b)
            return vector(words)
        target = nouns[0]
        mark('entity', *target.span())
        fronted = self.RELATION.search(text[:target.start()]) is not None
        relation = self.RELATION.search(text, target.end(), nouns[1].start()) if len(nouns) == 2 else None
        if relation and (not fronted) and (not re.search('\\b(between|among)\\b', text)):
            anchor = nouns[1]
            mark('relation', *relation.span())
            mark('anchor_entity', *anchor.span())
            clause(0, relation.start(), 'attribute', 'target_position')
            anchor_xy = clause(relation.end(), len(text), 'anchor_attribute', 'anchor_position')
            valid = bool(roles['rel_entity'].any() and roles['rel_anchor_entity'].any() and roles['rel_relation'].any())
            roles['rel_valid'] = torch.tensor(valid)
            if valid:
                words = set(relation.group().split())
                roles['rel_vector'] = vector(words)
                roles['rel_anchor_xy'] = anchor_xy
                roles['rel_size'] = torch.tensor(1.0 if words & {'bigger', 'larger', 'taller'} else -1.0 if words & {'smaller', 'shorter'} else 2.0 if 'similar' in words else 0.0)
        elif len(nouns) == 1 and (not fronted):
            roles['rel_target_xy'] = clause(0, len(text), 'attribute', 'target_position')
        if not roles['rel_entity'].any():
            for value in roles.values():
                value.zero_()
        return roles
