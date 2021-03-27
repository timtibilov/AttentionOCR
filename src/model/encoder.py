import torch
from .cnn import CNN
from typing import List
from torch import nn, Tensor
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    Combination of CNN and GRU together to encode image
    """

    def __init__(self, layers: List[int] = None, hidden_size: int = 128):
        """
        Parameters:
        layers:      list of size 3 of CNN block numbers on each layer
        hidden_size: size of embedding of feature vectors on each (h, w) position

        Note: encoding_size is 2 times greater 
        than hidden_size (encoding_size = 2 * hidden_size)
        """
        super().__init__()

        self.encoding_size = hidden_size * 2
        self.cnn = CNN(layers, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
        x: image with shape (1, channels, height, width)

        Returns:
        Tensor of shape (width * height, batch_size(=1), self.encoding_size)
        """
        x = self.cnn(x)
        # (width, height, channels)
        x = x.permute(2, 1, 0)
        x, _ = self.gru(x)
        # (height, width, hidden_size * 2)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(x.shape[0] * x.shape[1], 1, self.encoding_size)

        return x
