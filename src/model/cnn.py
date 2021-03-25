"""
Some of code was taken from 
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, List


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ):
        super().__init__()

        self.conv1 = conv3x3(input_size, output_size, stride)
        self.bn1 = norm_layer(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_size, output_size)
        self.bn2 = norm_layer(output_size)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNN(nn.Module):
    """
    Realizes ResNet-like neural network for one-dimentional pictures.
    """

    def __init__(
        self,
        layers: List[int],
        output_size: int = 128,
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.output = output_size
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(128, layers[0])
        self.layer2 = self._make_layer(256, layers[1])
        self.layer3 = self._make_layer(512, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.downsample = conv1x1(512, self.output)

    def _make_layer(self, output: int, blocks_num: int):
        
        pass

    def _make_block(self, output: int, stride: int = 2):
        pass