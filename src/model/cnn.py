"""
Some of code was taken from https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, List


def conv3x3(input_size: int, output_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(input_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(input_size: int, output_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(input_size, output_size, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.conv1 = conv3x3(input_size, output_size, stride)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_size, output_size)
        self.bn2 = nn.BatchNorm2d(output_size)
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
        self.input_size = 128
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, self.input_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.input_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1 = self._make_layer(128, layers[0])
        self.layer2 = self._make_layer(256, layers[1], stride=2)
        self.layer3 = self._make_layer(512, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.downsample = conv1x1(512, self.output)

    def _make_layer(self, output_size: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.input_size != output_size:
            downsample = nn.Sequential(
                conv1x1(self.input_size, output_size, stride),
                nn.BatchNorm2d(output_size),
            )
        layers = [BasicBlock(self.input_size, output_size, stride, downsample)]

        self.input_size = output_size

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.input_size, output_size))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.downsample(x)
        return x
