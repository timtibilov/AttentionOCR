import torch
from torch import nn, Tensor
from .encoder import Encoder
from .decoder import Decoder
from typing import Dict, Union
from torch.nn import functional as F


class AttentionOCR(nn.Module):

    def __init__(self, vocab_size: int, device: Union['cpu', 'cuda'] = 'cpu'):

        super().__init__()

        self.device = device
        self.encoder = Encoder([2, 3, 3])
        self.decoder = Decoder(vocab_size, self.encoder.encoding_size,
                               self.encoder.encoding_size, n_layers=2, dropout=0.2)
        self.vocab_size = vocab_size
        self.to(device)

    def forward(self, x: Dict[str, Union[Tensor, int]]) -> Tensor:
        hidden = decoder.h0
        output = decoder.SOS
        tokens = []
        for t in range(x['len']):
            seq = self.encoder(x['img'])
            logits, output, hidden = self.decoder(hidden, seq, output)
            tokens.append(F.softmax(logits))
        return torch.Tensor(tokens)
