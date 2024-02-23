import torch
from torch import nn
from typing import List, Union

class SSMultiscaleBlock3d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        output_prop_2: float = 0.5,
        output_prop_3: float = None,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        kernel_size_3: int = 3,
        stride: int = 1,
        dilation_1: Union[int, List[int]] = 1,
        dilation_2: Union[int, List[int]] = 1,
        dilation_3: Union[int, List[int]] = 1,
        groups: int = 1,
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.2),
        bias: bool = False,
        norm: str = 'batch',
        norm_dim: List[int] = None,
        norm_skip_first: bool = False,
        spectral_norm: bool = False
    ):
        super().__init__()

        output_channels_2 = round(output_channels * output_prop_2)
        output_channels_1 = output_channels - output_channels_2

        if isinstance(dilation_1, int):
            padding1 = (dilation_1 * (kernel_size_1 - 1)) // 2
        else:
            padding1 = list(map(lambda a: (a * (kernel_size_1 - 1)) // 2, dilation_1))

        if isinstance(dilation_2, int):
            padding2 = (dilation_2 * (kernel_size_2 - 1)) // 2
        else:
            padding2 = list(map(lambda a: (a * (kernel_size_2 - 1)) // 2, dilation_2))

        if output_prop_3 is not None and dilation_3 is not None:
            if isinstance(dilation_3, int):
                padding3 = (dilation_3 * (kernel_size_3 - 1)) // 2
            else:
                padding3 = map(lambda a: (a * (kernel_size_3 - 1)) // 2, dilation_3)

            output_channels_3 = round(output_channels * output_prop_3)
            output_channels_1 = output_channels - output_channels_2 - output_channels_3

        self.conv1 = nn.Conv3d(input_channels, output_channels_1, kernel_size=kernel_size_1, stride=stride, dilation=dilation_1, padding=padding1, groups=groups, bias=bias)
        self.conv2 = nn.Conv3d(output_channels_1, output_channels_2, kernel_size=kernel_size_2, stride=1, dilation=dilation_2, padding=padding2, groups=groups, bias=bias)
        self.conv3 = None if output_prop_3 is None else nn.Conv3d(output_channels_2, output_channels_3, kernel_size=kernel_size_3, stride=1, dilation=dilation_3, padding=padding3, groups=groups, bias=bias)

        if norm == 'instance':
            self.norm1 = None if norm_skip_first else nn.InstanceNorm3d(output_channels_1, affine=True)
            self.norm2 = nn.InstanceNorm3d(output_channels_2, affine=True)
            self.norm3 = None if output_prop_3 is None else nn.InstanceNorm3d(output_channels_3, affine=True)
        elif norm == 'layer':
            self.norm1 = None if norm_skip_first else nn.LayerNorm(norm_dim)
            self.norm2 = nn.LayerNorm(norm_dim)
            self.norm3 = None if output_prop_3 is None else nn.LayerNorm(norm_dim)
        elif norm == 'batch':
            self.norm1 = None if norm_skip_first else nn.BatchNorm3d(output_channels_1)
            self.norm2 = nn.BatchNorm3d(output_channels_2)
            self.norm3 = None if output_prop_3 is None else nn.BatchNorm3d(output_channels_3)
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

        self.activation = activation

        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = None if output_prop_3 is None else nn.utils.spectral_norm(self.conv3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []

        x = self.conv1(input)
        x = self.activation(x) if self.norm1 is None else self.activation(self.norm1(x))

        outputs.append(x)

        x = self.conv2(x)
        x = self.activation(x) if self.norm2 is None else self.activation(self.norm2(x))

        outputs.append(x)

        if self.conv3 is not None:
            x = self.conv3(x)
            x = self.activation(x) if self.norm3 is None else self.activation(self.norm3(x))

            outputs.append(x)

        return torch.cat(outputs, dim=1)