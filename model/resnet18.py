import torch.nn as nn
from torch import Tensor
from torch.nn.functional import relu


class Residual(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_channels: int, num_channels: int, use_1x1conv=False, strides=1) -> None:
        """
        Parameters:
        -------
        in_channels: number of channels of the input features
        num_channels: number of channels of the output features,
        use_1x1conv: whether to use 1x1conv to adjust channel of the output features,
        strides: strides of the ALL convolution layers in the class.
        """
        super().__init__()
        # f(x) - x
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3,
                      padding=1, stride=strides),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels,
                      kernel_size=3, padding=1,),
            nn.BatchNorm2d(num_channels),
        )
        # x
        self.use_1x1conv = use_1x1conv
        if use_1x1conv:
            self.adj_conv = nn.Conv2d(
                in_channels, num_channels, kernel_size=1, stride=strides)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        -------
        Parameters:
            x: the input tensor, should be [b,c,h,w] shape.
        """
        Y = self.features.forward(x)
        if self.use_1x1conv:
            x = self.adj_conv.forward(x)
        Y = Y + x
        return relu(Y)


def resnet_block(input_channels: int, num_channels: int, num_residuals: int, first_block=False) -> list[Residual]:
    """
    construct resnet bottleneck
    ------
    Parameters:
        in_channels: the channel number of input data
        num_channels: the channel number of output data
        num_residuals: the number of residual blocks
        first_block: if this block is the first block
    Return:
        blk:list[Residual]
    """
    blk = []
    for i in range(num_residuals):
        # if this block is the first block, we should double the number of channels
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                       use_1x1conv=True, strides=2))
        # else we don't need to double the number of channels, and we don't need 1x1 conv.
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class Resnet18(nn.Module):
    """
    Resnet18, with 18 weight layers.
    """

    def __init__(self, num_channels, num_classes) -> None:
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        b2 = nn.Sequential(*resnet_block(64, 64, 2, True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)
