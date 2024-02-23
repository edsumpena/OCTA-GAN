import torch
from torch import nn
import typing
from torch.nn import functional as F

import SingleShotMultiScale as ssms

class ConvPixelShuffleLite3d(nn.Module):
    """Three dimensional convolution with ICNR initialization followed by PixelShuffle.

    Increases `height` and `width` of `input` tensor by scale, acts like
    learnable upsampling. Due to `ICNR weight initialization <https://arxiv.org/abs/1707.02937>`__
    of `convolution` it has similar starting point to nearest neighbour upsampling.

    `kernel_size` got a default value of `3`, `upscale_factor` got a default
    value of `2`

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced after PixelShuffle
    upscale_factor : int, optional
        Factor to increase spatial resolution by. Default: `2`
    kernel_size : int or tuple, optional
        Size of the convolving kernel. Default: `3`
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
    initializer: typing.Callable[[torch.Tensor,], torch.Tensor], optional
        Initializer for ICNR initialization, can be a function from `torch.nn.init`.
        Gets and returns tensor after initialization.
        Default: `torch.nn.init.kaiming_normal_`

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels = None,
        upscale_factor: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: typing.Union[typing.Tuple[int, int], int, str] = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.2),
        norm: str = 'instance',
        spectral_norm: bool = False
    ):
        super().__init__()
        
        self.convolution1 = nn.Conv3d(
            in_channels,
            in_channels if hidden_channels is None else hidden_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
            padding_mode,
        )

        self.convolution2 = nn.Conv3d(
            in_channels if hidden_channels is None else hidden_channels,
            out_channels * upscale_factor * upscale_factor * upscale_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        if norm == 'instance':
            self.norm = nn.InstanceNorm3d(in_channels if hidden_channels is None else hidden_channels, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm3d(in_channels if hidden_channels is None else hidden_channels)
        
        self.activation = activation

        if spectral_norm:
            self.convolution1 = nn.utils.spectral_norm(self.convolution1)
            self.convolution2 = nn.utils.spectral_norm(self.convolution2)

        self.upsample = PixelShuffleNd(upscale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm(self.convolution1(inputs)))
        h = self.upsample(self.convolution2(h))
        return h

class SSMSConvPixelShuffleLite3d(nn.Module):
    """Three dimensional convolution with ICNR initialization followed by PixelShuffle.

    Increases `height` and `width` of `input` tensor by scale, acts like
    learnable upsampling. Due to `ICNR weight initialization <https://arxiv.org/abs/1707.02937>`__
    of `convolution` it has similar starting point to nearest neighbour upsampling.

    `kernel_size` got a default value of `3`, `upscale_factor` got a default
    value of `2`

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced after PixelShuffle
    upscale_factor : int, optional
        Factor to increase spatial resolution by. Default: `2`
    kernel_size : int or tuple, optional
        Size of the convolving kernel. Default: `3`
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
    initializer: typing.Callable[[torch.Tensor,], torch.Tensor], optional
        Initializer for ICNR initialization, can be a function from `torch.nn.init`.
        Gets and returns tensor after initialization.
        Default: `torch.nn.init.kaiming_normal_`

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels = None,
        upscale_factor: int = 2,
        output_prop_2: float = 0.5,
        output_prop_3: float = None,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        kernel_size_3: int = 3,
        dilation_1: typing.Union[int, typing.List[int]] = 1,
        dilation_2: typing.Union[int, typing.List[int]] = 2,
        dilation_3: typing.Union[int, typing.List[int]] = 3,
        groups: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.2),
        norm: str = 'instance',
        norm_dim: int = None,
        spectral_norm: bool = False
    ):
        super().__init__()
        
        self.ssms_conv1 = ssms.SSMultiscaleBlock3d(
            in_channels,
            (in_channels * 2) if hidden_channels is None else hidden_channels,
            output_prop_2=output_prop_2,
            output_prop_3=output_prop_3,
            kernel_size_1=kernel_size_1,
            kernel_size_2=kernel_size_2,
            kernel_size_3=kernel_size_3,
            stride=1,
            dilation_1=dilation_1,
            dilation_2=dilation_2,
            dilation_3=dilation_3,
            groups=groups,
            bias=False,
            activation=activation,
            norm=norm,
            norm_dim=norm_dim,
            spectral_norm=spectral_norm
        )

        self.convolution2 = nn.Conv3d(
            (in_channels * 2) if hidden_channels is None else hidden_channels,
            out_channels * upscale_factor * upscale_factor * upscale_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.activation = activation
        self.dilation_3_used = False if output_prop_3 is None else True

        if spectral_norm:
            self.convolution2 = nn.utils.spectral_norm(self.convolution2)

        self.upsample = PixelShuffleNd(upscale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.ssms_conv1(inputs)
        h = self.upsample(self.convolution2(h))
        return h


class PixelShuffleNd(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.
    Input Tensor must have at least 3 dimensions, e.g. :math:`(N, C, d_{1})` for 1D data,
    but Tensors with any number of dimensions after :math:`(N, C, ...)` (where N is mini-batch size,
    and C is channels) are supported.
    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details
    Args:
        upscale_factor (int): factor to increase spatial resolution by
    Shape:
        - Input: :math:`(N, C, d_{1}, d_{2}, ..., d_{n})`
        - Output: :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`
        Where :math:`n` is the dimensionality of the data, e.g. :math:`n-1` for 1D audio,
        :math:`n=2` for 2D images, etc.
    Examples::
        # 1D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffleNd, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return self.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
    
    def pixel_shuffle(self, input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
        r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
        tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
        Where :math:`n` is the dimensionality of the data.
        See :class:`~torch.nn.PixelShuffle` for details.
        Args:
            input (Variable): Input
            upscale_factor (int): factor to increase spatial resolution by
        Examples::
            # 1D example
            >>> input = torch.Tensor(1, 4, 8)
            >>> output = F.pixel_shuffle(input, 2)
            >>> print(output.size())
            torch.Size([1, 2, 16])
            # 2D example
            >>> input = torch.Tensor(1, 9, 8, 8)
            >>> output = F.pixel_shuffle(input, 3)
            >>> print(output.size())
            torch.Size([1, 1, 24, 24])
            # 3D example
            >>> input = torch.Tensor(1, 8, 16, 16, 16)
            >>> output = F.pixel_shuffle(input, 2)
            >>> print(output.size())
            torch.Size([1, 1, 32, 32, 32])
        """
        input_size = list(input.size())
        dimensionality = len(input_size) - 2

        input_size[1] //= (upscale_factor ** dimensionality)
        output_size = [dim * upscale_factor for dim in input_size[2:]]

        input_view = input.contiguous().view(
            input_size[0], input_size[1],
            *(([upscale_factor] * dimensionality) + input_size[2:])
        )

        indicies = list(range(2, 2 + 2 * dimensionality))
        indicies = indicies[1::2] + indicies[0::2]

        shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
        return shuffle_out.view(input_size[0], input_size[1], *output_size)