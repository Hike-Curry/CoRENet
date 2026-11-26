import copy
import json
import logging
import os
import warnings
from collections import UserDict
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from tokenizers import AddedToken
from tokenizers import Encoding as EncodingFast
from .file_utils import add_end_docstrings, cached_path, hf_bucket_url, is_remote_url, is_tf_available, is_torch_available, torch_required
if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch
logger = logging.getLogger(__name__)
VERY_LARGE_INTEGER = int(1e+30)
LARGE_INTEGER = int(1e+20)
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]
SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'
FULL_TOKENIZER_FILE = 'tokenizer.json'

class ExplicitEnum(Enum):

    @classmethod
    def _missing_(cls, value):
        raise ValueError('%r is not a valid %s, please select one of %s' % (value, cls.__name__, str(list(cls._value2member_map_.keys()))))

class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = 'only_first'
    ONLY_SECOND = 'only_second'
    LONGEST_FIRST = 'longest_first'
    DO_NOT_TRUNCATE = 'do_not_truncate'

class PaddingStrategy(ExplicitEnum):
    LONGEST = 'longest'
    MAX_LENGTH = 'max_length'
    DO_NOT_PAD = 'do_not_pad'

class TensorType(ExplicitEnum):
    PYTORCH = 'pt'
    TENSORFLOW = 'tf'
    NUMPY = 'np'

class CharSpan(NamedTuple):
    start: int
    end: int

class TokenSpan(NamedTuple):
    start: int
    end: int

class BatchEncoding(UserDict):

    def __init__(self, data: Optional[Dict[str, Any]]=None, encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]]=None, tensor_type: Union[None, str, TensorType]=None, prepend_batch_axis: bool=False):
        super().__init__(data)
        if isinstance(encoding, EncodingFast):
            encoding = [encoding]
        self._encodings = encoding
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def is_fast(self):
        return self._encodings is not None

    def __getitem__(self, item: Union[int, str]) -> EncodingFast:
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError('Indexing with integers (to access backend Encoding for a given batch index) is not available when using Python based tokenizers')

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {'data': self.data, 'encodings': self._encodings}

    def __setstate__(self, state):
        if 'data' in state:
            self.data = state['data']
        if 'encodings' in state:
            self._encodings = state['encodings']

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        return self._encodings

    def tokens(self, batch_index: int=0) -> List[str]:
        if not self._encodings:
            raise ValueError('tokens() is not available when using Python based tokenizers')
        return self._encodings[batch_index].tokens

    def words(self, batch_index: int=0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError('words() is not available when using Python based tokenizers')
        return self._encodings[batch_index].words

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int]=None) -> int:
        if not self._encodings:
            raise ValueError('token_to_word() is not available when using Python based tokenizers')
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(self, batch_or_word_index: int, word_index: Optional[int]=None) -> TokenSpan:
        if not self._encodings:
            raise ValueError('word_to_tokens() is not available when using Python based tokenizers')
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        return TokenSpan(*self._encodings[batch_index].word_to_tokens(word_index))

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int]=None) -> CharSpan:
        if not self._encodings:
            raise ValueError('token_to_chars() is not available when using Python based tokenizers')
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(*self._encodings[batch_index].token_to_chars(token_index))

    def char_to_token(self, batch_or_char_index: int, char_index: Optional[int]=None) -> int:
        if not self._encodings:
            raise ValueError('char_to_token() is not available when using Python based tokenizers')
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index)

    def word_to_chars(self, batch_or_word_index: int, word_index: Optional[int]=None) -> CharSpan:
        if not self._encodings:
            raise ValueError('word_to_chars() is not available when using Python based tokenizers')
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*self._encodings[batch_index].word_to_chars(word_index))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int]=None) -> int:
        if not self._encodings:
            raise ValueError('char_to_word() is not available when using Python based tokenizers')
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index)

    def convert_to_tensors(self, tensor_type: Union[None, str, TensorType], prepend_batch_axis: bool=False):
        if tensor_type is None:
            return self
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        if tensor_type == TensorType.TENSORFLOW and is_tf_available():
            as_tensor = tf.constant
        elif tensor_type == TensorType.PYTORCH and is_torch_available():
            as_tensor = torch.tensor
        elif tensor_type == TensorType.NUMPY:
            as_tensor = np.asarray
        else:
            raise ImportError('Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.'.format(tensor_type))
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]
                tensor = as_tensor(value)
                if tensor.ndim > 2:
                    tensor = tensor.squeeze(0)
                elif tensor.ndim < 2:
                    tensor = tensor[None, :]
                self[key] = tensor
            except:
                raise ValueError("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length.")
        return self

    @torch_required
    def to(self, device: str):
        self.data = {k: v.to(device) for k, v in self.data.items()}
        return self

class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additional_special_tokens']

    def __init__(self, verbose=True, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose
        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) and all((isinstance(t, str) for t in value))
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError('special token {} has to be either str or AddedToken but got: {}'.format(key, type(value)))

    def sanitize_special_tokens(self) -> int:
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]) -> int:
        if not special_tokens_dict:
            return 0
        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if self.verbose:
                logger.info('Assigning %s to the %s key of the tokenizer', value, key)
            setattr(self, key, value)
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all((isinstance(t, (str, AddedToken)) for t in value)), f'Tokens {value} for key {key} should all be str or AddedToken instances'
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(value, (str, AddedToken)), f'Token {value} for key {key} should be a str or an AddedToken instance'
                added_tokens += self.add_tokens([value], special_tokens=True)
        return added_tokens

    def add_tokens(self, new_tokens: Union[str, AddedToken, List[str], List[AddedToken]], special_tokens=False) -> int:
        if not new_tokens:
            return 0
        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    @property
    def bos_token(self):
        if self._bos_token is None and self.verbose:
            logger.error('Using bos_token, but it is not set yet.')
            return None
        return str(self._bos_token)

    @property
    def eos_token(self):
        if self._eos_token is None and self.verbose:
            logger.error('Using eos_token, but it is not set yet.')
            return None
        return str(self._eos_token)

    @property
    def unk_token(self):
        if self._unk_token is None and self.verbose:
            logger.error('Using unk_token, but it is not set yet.')
            return None
        return str(self._unk_token)

    @property
    def sep_token(self):
        if self._sep_token is None and self.verbose:
            logger.error('Using sep_token, but it is not set yet.')
            return None
        return str(self._sep_token)

    @property
    def pad_token(self):
        if self._pad_token is None and self.verbose:
            logger.error('Using pad_token, but it is not set yet.')
            return None
        return str(self._pad_token)

    @property
    def cls_token(self):
        if self._cls_token is None and self.verbose:
            logger.error('Using cls_token, but it is not set yet.')
            return None
        return str(self._cls_token)

    @property
    def mask_token(self):
        if self._mask_token is None and self.verbose:
            logger.error('Using mask_token, but it is not set yet.')
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self):
        if self._additional_special_tokens is None and self.verbose:
            logger.error('Using additional_special_tokens, but it is not set yet.')
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self):
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @property
    def special_tokens_map(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, '_' + attr)
            if attr_value:
                set_attr[attr] = str(attr_value)
        return set_attr

    @property
    def special_tokens_map_extended(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, '_' + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self):
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
ENCODE_KWARGS_DOCSTRING = "\n            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):\n                If set to ``True``, the sequences will be encoded with the special tokens relative\n                to their model.\n            `padding` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`False`):\n                Activate and control padding. Accepts the following values:\n\n                * `True` or `'longest'`: pad to the longest sequence in the batch (or no padding if only a single sequence if provided),\n                * `'max_length'`: pad to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`)\n                * `False` or `'do_not_pad'` (default): No padding (i.e. can output batch with sequences of uneven lengths)\n            `truncation` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`False`):\n                Activate and control truncation. Accepts the following values:\n\n                * `True` or `'longest_first'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`). This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided,\n                * `'only_first'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`). This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided,\n                * `'only_second'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`). This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided,\n                * `False` or `'do_not_truncate'` (default): No truncation (i.e. can output batch with sequences length greater than the model max admissible input size)\n            `max_length` (:obj:`Union[int, None]`, `optional`, defaults to :obj:`None`):\n                Control the length for padding/truncation. Accepts the following values\n\n                * `None` (default): This will use the predefined model max length if required by one of the truncation/padding parameters. If the model has no specific max input length (e.g. XLNet) truncation/padding to max length is deactivated.\n                * `any integer value` (e.g. `42`): Use this specific maximum length value if required by one of the truncation/padding parameters.\n            stride (:obj:`int`, `optional`, defaults to ``0``):\n                If set to a number along with max_length, the overflowing tokens returned when `return_overflowing_tokens=True`\n                will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflow ing sequences.\n                The value of this argument defines the number of overlapping tokens.\n            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):\n                Set to True to indicate the input is already tokenized\n            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.\n                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability\n                >= 7.5 (Volta).\n            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):\n                Can be set to 'tf', 'pt' or 'np' to return respectively TensorFlow :obj:`tf.constant`,\n                PyTorch :obj:`torch.Tensor` or Numpy :oj: `np.ndarray` instead of a list of python integers.\n"
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = "\n            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):\n                Whether to return token type IDs. If left to the default, will return the token type IDs according\n                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.\n\n                `What are token type IDs? <../glossary.html#token-type-ids>`_\n            return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`none`):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.\n\n                `What are attention masks? <../glossary.html#attention-mask>`__\n            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):\n                Set to True to return overflowing token sequences (default False).\n            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):\n                Set to True to return special tokens mask information (default False).\n            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):\n                Set to True to return (char_start, char_end) for each token (default False).\n                If using Python's tokenizer, this method will raise NotImplementedError.\n                This one is only available on fast tokenizers inheriting from PreTrainedTokenizerFast.\n            **kwargs: passed to the `self.tokenize()` method\n\n        Return:\n            A Dictionary of shape::\n\n                {\n                    input_ids: list[int],\n                    token_type_ids: list[int] if return_token_type_ids is True (default)\n                    attention_mask: list[int] if return_attention_mask is True (default)\n                    overflowing_tokens: list[int] if the tokenizer is a slow tokenize, else a List[List[int]] if a ``max_length`` is specified and ``return_overflowing_tokens=True``\n                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``\n                    and return_special_tokens_mask is True\n                }\n\n            With the fields:\n\n            - ``input_ids``: list of token ids to be fed to a model\n            - ``token_type_ids``: list of token type ids to be fed to a model\n            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model\n            - ``overflowing_tokens``: list of overflowing tokens sequences if a max length is specified and ``return_overflowing_tokens=True``.\n            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added\n              tokens and 1 specifying sequence tokens.\n"

class PreTrainedTokenizerBase(SpecialTokensMixin):
    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, int] = {}
    model_input_names: List[str] = ['token_type_ids', 'attention_mask']
    padding_side: str = 'right'

    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = kwargs
        model_max_length = kwargs.pop('model_max_length', kwargs.pop('max_len', None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.padding_side = kwargs.pop('padding_side', self.padding_side)
        assert self.padding_side in ['right', 'left'], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
        self.model_input_names = kwargs.pop('model_input_names', self.model_input_names)
        super().__init__(**kwargs)

    @property
    def max_len(self) -> int:
        return self.model_max_length

    @property
    def max_len_single_sentence(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self) -> int:
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            logger.warning("Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up.")
        else:
            raise ValueError("Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up.")

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            logger.warning("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")
        else:
            raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if cls.pretrained_init_configuration and pretrained_model_name_or_path in cls.pretrained_init_configuration:
                init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path].copy()
        else:
            logger.info("Model name '{}' not found in model shortcut name list ({}). Assuming '{}' is a path, a model identifier, or url to a directory containing tokenizer files.".format(pretrained_model_name_or_path, ', '.join(s3_models), pretrained_model_name_or_path))
            if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                if len(cls.vocab_files_names) > 1:
                    raise ValueError('Calling {}.from_pretrained() with the path to a single file or url is not supported.Use a model identifier or the path to a directory instead.'.format(cls.__name__))
                logger.warning('Calling {}.from_pretrained() with the path to a single file or url is deprecated'.format(cls.__name__))
                file_id = list(cls.vocab_files_names.keys())[0]
                vocab_files[file_id] = pretrained_model_name_or_path
            else:
                additional_files_names = {'added_tokens_file': ADDED_TOKENS_FILE, 'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE, 'tokenizer_config_file': TOKENIZER_CONFIG_FILE, 'full_tokenizer_file': FULL_TOKENIZER_FILE}
                for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
                    if os.path.isdir(pretrained_model_name_or_path):
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                        if not os.path.exists(full_file_name):
                            logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                            full_file_name = None
                    else:
                        full_file_name = hf_bucket_url(pretrained_model_name_or_path, filename=file_name, use_cdn=False)
                    vocab_files[file_id] = full_file_name
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(file_path, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only)
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                msg = "Couldn't reach server at '{}' to download vocabulary files."
            else:
                msg = "Model name '{}' was not found in tokenizers model name list ({}). We assumed '{}' was a path or url to a directory containing vocabulary files named {}, but couldn't find such vocabulary files at this path or url.".format(pretrained_model_name_or_path, ', '.join(s3_models), pretrained_model_name_or_path, list(cls.vocab_files_names.values()))
            raise EnvironmentError(msg)
        if all((full_file_name is None for full_file_name in resolved_vocab_files.values())):
            raise EnvironmentError("Model name '{}' was not found in tokenizers model name list ({}). We assumed '{}' was a path, a model identifier, or url to a directory containing vocabulary files named {} but couldn't find such vocabulary files at this path or url.".format(pretrained_model_name_or_path, ', '.join(s3_models), pretrained_model_name_or_path, list(cls.vocab_files_names.values())))
        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info('loading file {}'.format(file_path))
            else:
                logger.info('loading file {} from cache at {}'.format(file_path, resolved_vocab_files[file_id]))
        tokenizer_config_file = resolved_vocab_files.pop('tokenizer_config_file', None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding='utf-8') as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            saved_init_inputs = init_kwargs.pop('init_inputs', ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration
        init_kwargs.update(kwargs)
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kwargs['model_max_length'] = min(init_kwargs.get('model_max_length', int(1e+30)), model_max_length)
        added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError('Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.')
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs
        special_tokens_map_file = resolved_vocab_files.pop('special_tokens_map_file', None)
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding='utf-8') as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if isinstance(value, dict):
                    value = AddedToken(**value)
                setattr(tokenizer, key, value)
        special_tokens = tokenizer.all_special_tokens
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding='utf-8') as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)
            added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(), key=lambda x: x[1]))
            for token, index in added_tok_encoder_sorted:
                assert index == len(tokenizer), f"Non-consecutive added token '{token}' found. Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
                tokenizer.add_tokens(token, special_tokens=bool(token in special_tokens))
        added_tokens = tokenizer.sanitize_special_tokens()
        if added_tokens:
            logger.warning('Special tokens have been added in the vocabulary, make sure the associated word emebedding are fine-tuned or trained.')
        return tokenizer

    def save_pretrained(self, save_directory) -> Tuple[str]:
        if os.path.isfile(save_directory):
            logger.error('Provided path ({}) should be a directory, not a file'.format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)
        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)
        with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))
        with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
            write_dict = {}
            for key, value in self.special_tokens_map_extended.items():
                if isinstance(value, AddedToken):
                    write_dict[key] = value.__getstate__()
                else:
                    write_dict[key] = value
            f.write(json.dumps(write_dict, ensure_ascii=False))
        added_vocab = self.get_added_vocab()
        if added_vocab:
            with open(added_tokens_file, 'w', encoding='utf-8') as f:
                out_str = json.dumps(added_vocab, ensure_ascii=False)
                f.write(out_str)
        vocab_files = self.save_vocabulary(save_directory)
        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str]=False, truncation: Union[bool, str]=False, max_length: Optional[int]=None, stride: int=0, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs):
        encoded_inputs = self.encode_plus(text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, return_tensors=return_tensors, **kwargs)
        return encoded_inputs['input_ids']

    def num_special_tokens_to_add(self, pair: bool=False) -> int:
        raise NotImplementedError

    def _get_padding_truncation_strategies(self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs):
        old_truncation_strategy = kwargs.pop('truncation_strategy', 'do_not_truncate')
        old_pad_to_max_length = kwargs.pop('pad_to_max_length', False)
        if max_length is not None and padding is False and (truncation is False):
            if verbose:
                logger.warning("Truncation was not explicitely activated but `max_length` is provided a specific value, please use `truncation=True` to explicitely truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.")
            truncation = 'longest_first'
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn("The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).", DeprecationWarning)
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD
        if truncation is False and old_truncation_strategy != 'do_not_truncate':
            if verbose:
                warnings.warn("The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).", DeprecationWarning)
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = TruncationStrategy.LONGEST_FIRST
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        logger.warning('Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no padding.')
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        logger.warning('Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.')
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad_token or self.pad_token_id < 0):
            raise ValueError("Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.")
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and padding_strategy != PaddingStrategy.DO_NOT_PAD and (pad_to_multiple_of is not None) and (max_length is not None) and (max_length % pad_to_multiple_of != 0):
            raise ValueError(f'Truncation and padding are both activated but truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of}).')
        return (padding_strategy, truncation_strategy, max_length, kwargs)

    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str]=False, truncation: Union[bool, str]=False, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        assert isinstance(text, str) or (isinstance(text, (list, tuple)) and (len(text) == 0 or (isinstance(text[0], str) or (isinstance(text[0], (list, tuple)) and (len(text[0]) == 0 or isinstance(text[0][0], str)))))), 'text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).'
        assert text_pair is None or isinstance(text_pair, str) or (isinstance(text_pair, (list, tuple)) and (len(text_pair) == 0 or (isinstance(text_pair[0], str) or (isinstance(text_pair[0], (list, tuple)) and (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str)))))), 'text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).'
        is_batched = bool(not is_pretokenized and isinstance(text, (list, tuple)) or (is_pretokenized and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))))
        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, is_pretokenized=is_pretokenized, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        else:
            return self.encode_plus(text=text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, is_pretokenized=is_pretokenized, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str]=False, truncation: Union[bool, str]=False, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        return self._encode_plus(text=text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, is_pretokenized=is_pretokenized, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        raise NotImplementedError

    def batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]], add_special_tokens: bool=True, padding: Union[bool, str]=False, truncation: Union[bool, str]=False, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        return self._batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, is_pretokenized=is_pretokenized, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_pretokenized: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        raise NotImplementedError

    def pad(self, encoded_inputs: Union[BatchEncoding, List[BatchEncoding], Dict[str, EncodedInput], Dict[str, List[EncodedInput]], List[Dict[str, EncodedInput]]], padding: Union[bool, str]=True, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None, verbose: bool=True) -> BatchEncoding:
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
        assert 'input_ids' in encoded_inputs, 'You should supply an encoding or a list of encodings to this method. An encoding is the output of one the encoding methods of the tokenizer, i.e. __call__/encode_plus/batch_encode_plus. '
        if not encoded_inputs['input_ids']:
            if return_attention_mask:
                encoded_inputs['attention_mask'] = []
            return encoded_inputs
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(padding=padding, max_length=max_length, verbose=verbose)
        if encoded_inputs['input_ids'] and (not isinstance(encoded_inputs['input_ids'][0], (list, tuple))):
            encoded_inputs = self._pad(encoded_inputs, max_length=max_length, padding_strategy=padding_strategy, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
        batch_size = len(encoded_inputs['input_ids'])
        assert all((len(v) == batch_size for v in encoded_inputs.values())), 'Some items in the output dictionnary have a different batch size than others.'
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max((len(inputs) for inputs in encoded_inputs['input_ids']))
            padding_strategy = PaddingStrategy.MAX_LENGTH
        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict(((k, v[i]) for k, v in encoded_inputs.items()))
            outputs = self._pad(inputs, max_length=max_length, padding_strategy=padding_strategy, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(self, token_ids_0: List, token_ids_1: Optional[List]=None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List]=None) -> List:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def prepare_for_model(self, ids: List[int], pair_ids: Optional[List[int]]=None, add_special_tokens: bool=True, padding: Union[bool, str]=False, truncation: Union[bool, str]=False, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, prepend_batch_axis: bool=False, **kwargs) -> BatchEncoding:
        if 'return_lengths' in kwargs:
            if verbose:
                warnings.warn('The PreTrainedTokenizerBase.prepare_for_model `return_lengths` parameter is deprecated. Please use `return_length` instead.', FutureWarning)
            return_length = kwargs['return_lengths']
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        if return_token_type_ids is None:
            return_token_type_ids = 'token_type_ids' in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        encoded_inputs = {}
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and (total_len > max_length):
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids, pair_ids=pair_ids, num_tokens_to_remove=total_len - max_length, truncation_strategy=truncation_strategy, stride=stride)
            if return_overflowing_tokens:
                encoded_inputs['overflowing_tokens'] = overflowing_tokens
                encoded_inputs['num_truncated_tokens'] = total_len - max_length
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])
        encoded_inputs['input_ids'] = sequence
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs['special_tokens_mask'] = [0] * len(sequence)
        if max_length is None and len(encoded_inputs['input_ids']) > self.model_max_length and verbose:
            logger.warning('Token indices sequence length is longer than the specified maximum sequence length for this model ({} > {}). Running this sequence through the model will result in indexing errors'.format(len(ids), self.model_max_length))
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(encoded_inputs, max_length=max_length, padding=padding_strategy.value, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        if return_length:
            encoded_inputs['length'] = len(encoded_inputs['input_ids'])
        batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)
        return batch_outputs

    def truncate_sequences(self, ids: List[int], pair_ids: Optional[List[int]]=None, num_tokens_to_remove: int=0, truncation_strategy: Union[str, TruncationStrategy]='longest_first', stride: int=0) -> Tuple[List[int], List[int], List[int]]:
        if num_tokens_to_remove <= 0:
            return (ids, pair_ids, [])
        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)
        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if not overflowing_tokens:
                        window_len = min(len(ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids[-window_len:])
                    ids = ids[:-1]
                else:
                    if not overflowing_tokens:
                        window_len = min(len(pair_ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(pair_ids[-window_len:])
                    pair_ids = pair_ids[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]
            else:
                logger.error(f"We need to remove {num_tokens_to_remove} to truncate the inputbut the first sequence has a length {len(ids)}. Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_second'.")
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                overflowing_tokens = pair_ids[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
            else:
                logger.error(f"We need to remove {num_tokens_to_remove} to truncate the inputbut the second sequence has a length {len(pair_ids)}. Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_first'.")
        return (ids, pair_ids, overflowing_tokens)

    def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs['input_ids'])
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_inputs['input_ids']) != max_length
        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs['input_ids'])
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [1] * len(encoded_inputs['input_ids']) + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'] + [self.pad_token_type_id] * difference
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
                encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [0] * difference + [1] * len(encoded_inputs['input_ids'])
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [self.pad_token_type_id] * difference + encoded_inputs['token_type_ids']
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [1] * difference + encoded_inputs['special_tokens_mask']
                encoded_inputs['input_ids'] = [self.pad_token_id] * difference + encoded_inputs['input_ids']
            else:
                raise ValueError('Invalid padding strategy:' + str(self.padding_side))
        elif return_attention_mask:
            encoded_inputs['attention_mask'] = [1] * len(encoded_inputs['input_ids'])
        return encoded_inputs

    def batch_decode(self, sequences: List[List[int]], **kwargs) -> List[str]:
        return [self.decode(seq, **kwargs) for seq in sequences]

    def decode(self, token_ids: List[int], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=True) -> str:
        raise NotImplementedError

    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List]=None, already_has_special_tokens: bool=False) -> List[int]:
        assert already_has_special_tokens and token_ids_1 is None, 'You cannot use ``already_has_special_tokens=False`` with this tokenizer. Please use a slow (full python) tokenizer to activate this argument.Or set `return_special_token_mask=True` when calling the encoding method to get the special tokens mask in any tokenizer. '
        all_special_ids = self.all_special_ids
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        return out_string
