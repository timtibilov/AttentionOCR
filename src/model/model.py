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
        self.encoder = Encoder([2, 2, 2])
        self.decoder = Decoder(vocab_size, self.encoder.encoding_size,
                               self.encoder.encoding_size, n_layers=2, dropout=0.2)
        self.vocab_size = vocab_size
        self.softmax = nn.LogSoftmax(dim=1)
        self.to(device)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        hidden = self.decoder.h0
        seq = self.encoder(x['img'])
        tokens = torch.Tensor().to(self.device)
        for t in range(x['len']):
            logits, hidden = self.decoder(hidden, seq)
            tokens = torch.cat((tokens, self.softmax(logits)), dim=0)
        return tokens

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(
            path, map_location=torch.device('cpu')))
