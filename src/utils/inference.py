import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from loguru import logger
from torchvision.transforms import ToTensor
from typing import Union, Dict, Optional, Tuple
from werkzeug.datastructures import FileStorage
sys.path.insert(0, f'{os.path.join(os.path.dirname(__file__), "../")}')
from model.model import AttentionOCR


def load_vocab(vocab_path: str) -> Tuple[Dict[int, str], int]:
    with open(vocab_path, 'r') as f:
        strs = f.readlines()
    vocab = {i: s.strip() for i, s in enumerate(strs)}
    reversed_vocab = {vocab[k]: k for k in vocab}

    if 'EOF' not in reversed_vocab:
        raise ValueError('There is no EOF token in vocabulary')
    eof_index = reversed_vocab['EOF']

    return vocab, eof_index


def process_img(
    img: Union[str, FileStorage],
    transform: Optional[ToTensor] = None
) -> Tensor:

    img = Image.open(img).convert('L')

    if transform:
        img = transform(img)
        img = img.unsqueeze(0)

    img[img > 0.77] = 1.

    return img


class ModelManager(object):

    def __new__(
        cls,
        model_path: str,
        vocab_path: str,
        max_len: int,
        device: Union['cpu', 'cuda'] = 'cpu'
    ):
        if not hasattr(cls, 'instance'):
            cls.__vocab, eof_index = load_vocab(vocab_path)
            cls.__model = AttentionOCR(len(cls.__vocab), device, max_len, eof_index)
            cls.__model.load(model_path)
            cls.__model.to(device)
            cls.__device = device
            cls.__transform = ToTensor()
            cls.instance = super(ModelManager, cls).__new__(cls)
        return cls.instance

    def to(self, device: Union['cpu', 'cuda']):
        try:
            self.__model.to(device)
            self.__device = device
        except Exception as e:
            logger.exception(
                f'Tried to move on {device} device, got exception {e}')

    def predict(self, img: Union[str, FileStorage]) -> str:
        img = process_img(img, self.__transform)
        encoded_tokens = self.__model.inference(img)
        tokens = [self.__vocab[t] for t in encoded_tokens]
        return ''.join(tokens)
