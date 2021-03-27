import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    """
    Implementation of Bahdanau visual attention.
    See https://arxiv.org/pdf/1409.0473.pdf for more information
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.downsample = nn.Linear(n_layers, 1)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.features_fc = nn.Linear(input_size, hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)

    def downsample_hiddens(self, hidden_state: Tensor) -> Tensor:
        # (batch_size, hidden_size, n_layers)
        hidden_state = hidden_state.permute(1, 2, 0)
        hidden_state = self.downsample(hidden_state)
        # (1, batch_size, hidden_size)
        hidden_state = hidden_state.permute(2, 0, 1)
        return hidden_state

    def forward(self, hidden_state: Tensor, features: Tensor) -> Tensor:
        """
        Parameters:
        hidden_state: a tensor with shape (n_layers, batch_size(=1), hidden_size)
        features: a tensor of image encoded features with size (seq_len, batch_size, encoding_size)

        Returns:
        Tensor of shape (1, batch_size(=1), encoding_size)
        """
        if self.n_layers > 1:
            hidden_state = self.downsample_hiddens(hidden_state)

        # (seq_len, 1, 1)
        scores = self.scorer(F.tanh(
            self.features_fc(features) + self.hidden_fc(hidden_state)))

        weights = F.softmax(scores, dim=0)
        # (1, 1, input_size)
        decoder_input = torch.sum(features * weights, dim=0).unsqueeze(0)
        return decoder_input
