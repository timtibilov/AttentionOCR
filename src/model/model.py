import torch
from torch import nn, Tensor
from .encoder import Encoder
from .decoder import Decoder
from .cnn import CNN, ResNetCNN
from torch.nn import functional as F
from typing import Dict, Union, Optional, List


class AttentionOCR(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        device: Union['cpu', 'cuda'] = 'cpu',
        max_len: int = 150,
        eof_index: Optional[int] = None,
        cnn_type: Union[CNN, ResNetCNN] = ResNetCNN,
    ):

        super().__init__()

        self.device = device
        if not 0 < max_len <= 300:
            raise ValueError(f'max_len must be between 1 and 300, got {max_len}')
        if eof_index is not None and eof_index < 0:
            raise ValueError(f'EOF token index must be zero or positive, got {eof_index}')
        self.eof = eof_index
        self.max_len = max_len
        self.encoder = Encoder(cnn=cnn_type)
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

    @torch.no_grad()
    def inference(self, x: Tensor) -> List[int]:
        self.eval()
        hidden = self.decoder.h0
        seq = self.encoder(x)
        tokens = []
        while len(tokens) < self.max_len:
            logits, hidden = self.decoder(hidden, seq)
            token = torch.argmax(self.softmax(logits), -1)
            if token == self.eof:
                return tokens
            tokens.append(token.detach().cpu().item())
        return tokens

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(
            path, map_location=torch.device('cpu')))
