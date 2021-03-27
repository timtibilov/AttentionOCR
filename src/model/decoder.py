import torch
from typing import Tuple
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
        self.output_fc = nn.Linear(encoding_size + hidden_size, hidden_size)
        self.decode_fc = nn.Linear(hidden_size, vocab_size)
        self.SOS = nn.Parameter(torch.rand(
            (n_layers, 1, encoding_size)), requires_grad=True)
        self.h0 = nn.Parameter(torch.randn(
            (n_layers, 1, hidden_size)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        prev_hidden: Tensor,
        encoded_seq: Tensor,
        prev_output: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:

        output, hidden = self.gru(prev_output, prev_hidden)
        context = self.attention(hidden, encoded_seq)
        output = F.tanh(self.output_fc(torch.cat((context, output), dim=-1)))
        logits = self.decode_fc(self.dropout(output))

        return logits.squeeze(), output, hidden

    def another_forward(
        self,
        prev_hidden: Tensor,
        encoded_seq: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        context = self.attention(prev_hidden, encoded_seq)
        output, hidden = self.gru(context, prev_hidden)
        output = F.tanh(self.output_fc(torch.cat((context, output), dim=-1)))
        logits = self.decode_fc(self.dropout(output))

        return logits, hidden
