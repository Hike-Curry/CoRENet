import itertools
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from .file_utils import add_end_docstrings
from .tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING, AddedToken, BatchEncoding, EncodedInput, EncodedInputPair, PaddingStrategy, PreTokenizedInput, PreTokenizedInputPair, PreTrainedTokenizerBase, TensorType, TextInput, TextInputPair, TruncationStrategy
logger = logging.getLogger(__name__)

def _is_whitespace(char):
    if char == ' ' or char == '\t' or char == '\n' or (char == '\r'):
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False

def _is_control(char):
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False

def _is_punctuation(char):
    cp = ord(char)
    if cp >= 33 and cp <= 47 or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False

def _is_end_of_word(text):
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))

def _is_start_of_word(text):
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))

class PreTrainedTokenizer(PreTrainedTokenizerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def get_vocab(self):
        raise NotImplementedError()

    def get_added_vocab(self) -> Dict[str, int]:
        return self.added_tokens_encoder

    def __len__(self):
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens=False) -> int:
        new_tokens = [str(tok) for tok in new_tokens]
        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and self.init_kwargs.get('do_lower_case', False):
                token = token.lower()
            if token != self.unk_token and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token) and (token not in tokens_to_add):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info('Adding %s to the vocabulary', token)
        added_tok_encoder = dict(((tok, len(self) + i) for i, tok in enumerate(tokens_to_add)))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)
        if special_tokens:
            self.unique_no_split_tokens = list(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            self.unique_no_split_tokens = list(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        return len(tokens_to_add)

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: TextInput, **kwargs):
        all_special_tokens_extended = dict(((str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)))
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)
        if kwargs:
            logger.warning(f'Keyword arguments {kwargs} not recognized.')
        if self.init_kwargs.get('do_lower_case', False):
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ''
            for i, sub_text in enumerate(split_text):
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        if i < len(split_text) - 1 and (not _is_end_of_word(sub_text)) and (not _is_start_of_word(split_text[i + 1])):
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result += [full_word]
                            full_word = ''
                            continue
                    if tok_extended.rstrip and i > 0:
                        sub_text = sub_text.lstrip()
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                else:
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()
                if i == 0 and (not sub_text):
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text
            return list(itertools.chain.from_iterable((self._tokenize(token) if token not in self.unique_no_split_tokens else [token] for token in tokenized_text)))
        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_pretokenized:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_pretokenized=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            elif is_pretokenized:
                raise ValueError(f'Input {text} is not valid. Should be a string or a list/tuple of strings when `is_pretokenized=True`.')
            else:
                raise ValueError(f'Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers.To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.More information on available tokenizers at https://github.com/huggingface/transformers/pull/2674')
        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        return self.prepare_for_model(first_ids, pair_ids=second_ids, add_special_tokens=add_special_tokens, padding=padding_strategy.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, prepend_batch_axis=True, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, verbose=verbose)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_pretokenized:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_pretokenized=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError('Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers.To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.')
        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = (ids_or_pair_ids, None)
            elif is_pretokenized and (not isinstance(ids_or_pair_ids[0], (list, tuple))):
                ids, pair_ids = (ids_or_pair_ids, None)
            else:
                ids, pair_ids = ids_or_pair_ids
            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))
        batch_outputs = self._batch_prepare_for_model(input_ids, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=return_tensors, verbose=verbose)
        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(self, batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(first_ids, second_ids, add_special_tokens=add_special_tokens, padding=PaddingStrategy.DO_NOT_PAD.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=None, return_attention_mask=False, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=None, prepend_batch_axis=False, verbose=verbose)
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = self.pad(batch_outputs, padding=padding_strategy.value, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    def prepare_for_tokenization(self, text: str, is_pretokenized=False, **kwargs) -> (str, dict):
        return (text, kwargs)

    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List]=None, already_has_special_tokens: bool=False) -> List[int]:
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool=False) -> Union[str, List[str]]:
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ' '.join(self.convert_ids_to_tokens(tokens))

    def decode(self, token_ids: List[int], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=True) -> str:
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ' '.join(sub_texts)
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory) -> Tuple[str]:
        raise NotImplementedError
