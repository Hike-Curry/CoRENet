import re
import torch
from transformers import BertTokenizerFast
from data.text import RoleParser
from data.refsegrs import CLASSES

class CompositionRoles:
    NAMES = ('object', 'modifier', 'anchor', 'anchor_modifier', 'relation', 'position')
    REL = re.compile('\\b(?:(?:(?:upper|lower|top|bottom)\\s+)?(?:left|right|north|south|east|west)\\s+of|along\\s+with|surrounded\\s+by|next\\s+to|adjacent\\s+to|above|below|near|beside|along|inside|outside|around|on|in|at)\\b')
    POS = set('left right top bottom upper lower north south east west center central middle leftmost rightmost'.split())
    STOP = set('a an the is are was were be being of on in at to from and or with that which located situated side part corner'.split())

    def __init__(self):
        self.nouns = RoleParser(list(CLASSES) + ['parking area', 'parking lot']).nouns

    def __call__(self, text, offsets):
        text = text.lower()
        result = {'comp_' + key: torch.zeros(len(offsets)) for key in self.NAMES}
        nouns = list(self.nouns.finditer(text))
        if not nouns:
            return result

        def mark(key, start, end):
            for i, (a, b) in enumerate(offsets):
                if b > a and a < end and (b > start):
                    result['comp_' + key][i] = 1
        if self.REL.search(text[:nouns[0].start()]) or re.search('\\b(between|among)\\b', text):
            return result
        head = nouns[0]
        mark('object', *head.span())
        relation = self.REL.search(text, head.end(), nouns[1].start()) if len(nouns) == 2 else None

        def modifiers(start, end, key, allow_position):
            for word in re.finditer('\\b[\\w-]+\\b', text[start:end]):
                a, b = (start + word.start(), start + word.end())
                if any((a < noun.end() and b > noun.start() for noun in nouns)):
                    continue
                if word.group() in self.POS and allow_position:
                    mark('position', a, b)
                elif word.group() not in self.STOP:
                    mark(key, a, b)
        if relation:
            mark('relation', *relation.span())
            mark('anchor', *nouns[1].span())
            modifiers(0, relation.start(), 'modifier', True)
            modifiers(relation.end(), len(text), 'anchor_modifier', False)
            if not result['comp_anchor'].any():
                for key in ('anchor', 'anchor_modifier', 'relation'):
                    result['comp_' + key].zero_()
        elif len(nouns) == 1:
            modifiers(0, len(text), 'modifier', True)
        else:
            modifiers(0, head.end(), 'modifier', True)
        if not result['comp_object'].any():
            for value in result.values():
                value.zero_()
        return result

def add_roles(expressions, bert_dir):
    tokenizer = BertTokenizerFast.from_pretrained(bert_dir, local_files_only=True, clean_up_tokenization_spaces=False)
    parser, cache = (CompositionRoles(), {})
    for expression in expressions:
        text = expression['text']
        if text not in cache:
            n = len(expression['ids'])
            encoding = tokenizer(text, return_offsets_mapping=True)
            offsets = encoding['offset_mapping'][:n]
            offsets += [(0, 0)] * (n - len(offsets))
            cache[text] = parser(text, offsets)
        expression['roles'] = cache[text]
