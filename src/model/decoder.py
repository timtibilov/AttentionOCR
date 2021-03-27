import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .attention import BahdanauAttention


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        encoding_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.
    ):
        super().__init__()

        if n_layers == 1 and dropout != 0:
            raise ValueError(
                f'Got n_layers {n_layers} (!= 1) and dropout != 0')
        if not 0 <= dropout < 1:
            raise ValueError(
                f'Dropout must be in interval [0; 1), got {dropout}')

        self.vocab_size = vocab_size
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.attention = BahdanauAttention(
            encoding_size, hidden_size, n_layers)
        self.gru = nn.GRU(encoding_size, hidden_size,
                          num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(encoding_size + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size)
        self.SOS = nn.Parameter(torch.rand(
            (n_layers, 1, encoding_size)), requires_grad=True)
        self.h0 = nn.Parameter(torch.randn(
            (n_layers, 1, hidden_size)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prev_hidden: Tensor, encoded_seq: Tensor, prev_output: Tensor) -> Tensor:
        pass
