import torch
from torch import nn
from torch.nn import functional as F
from typing import List

import PixelShuffle as pixOp
import SingleShotMultiScale as ssms  

class Generator(nn.Module):
    def __init__(self, noise:int=1024, channels:List[int]=[512, 512, 256, 128, 64, 32]):
        super(Generator, self).__init__()

        scale_step = 2
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()
        self.noise = noise

        self.tp_conv1 = nn.utils.spectral_norm(nn.ConvTranspose3d(noise, channels[0], kernel_size=4, stride=1, padding=0, bias=False))
        self.in1 = nn.InstanceNorm3d(channels[0], affine=True)

        self.resize_conv1 = nn.utils.spectral_norm(nn.Conv3d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False))
        self.ssms_conv_pix_shuff1 = pixOp.SSMSConvPixelShuffleLite3d(channels[0], channels[1], upscale_factor=scale_step, output_prop_2=0.3, output_prop_3=0.2, kernel_size_1=3, kernel_size_2=3, kernel_size_3=3, dilation_1=(1, 1, 1), dilation_2=(1, 1, 1), dilation_3=(1, 1, 1), activation=self.activation, norm='instance', spectral_norm=True)
        self.in2 = nn.InstanceNorm3d(channels[1], affine=True)
        
        self.resize_conv2 = nn.utils.spectral_norm(nn.Conv3d(channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_pix_shuff2 = pixOp.ConvPixelShuffleLite3d(channels[1], channels[2], upscale_factor=scale_step, padding=1, activation=self.activation, norm='instance', spectral_norm=True)
        self.in3 = nn.InstanceNorm3d(channels[2], affine=True)
        
        self.resize_conv3 = nn.utils.spectral_norm(nn.Conv3d(channels[2], channels[3], kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_pix_shuff3 = pixOp.ConvPixelShuffleLite3d(channels[2], channels[3], upscale_factor=scale_step, activation=self.activation, norm='instance', spectral_norm=True)
        self.in4 = nn.InstanceNorm3d(channels[3], affine=True)
        
        self.resize_conv4 = nn.utils.spectral_norm(nn.Conv3d(channels[3], channels[4], kernel_size=1, stride=1, padding=0, bias=False))
        self.ssms_conv_pix_shuff4 = pixOp.SSMSConvPixelShuffleLite3d(channels[3], channels[4], upscale_factor=scale_step, output_prop_2=0.3, output_prop_3=0.2, kernel_size_1=3, kernel_size_2=3, kernel_size_3=3, dilation_1=(1, 1, 1), dilation_2=(1, 1, 1), dilation_3=(1, 1, 1), activation=self.activation, norm='instance', spectral_norm=True)
        self.in5 = nn.InstanceNorm3d(channels[4], affine=True)

        self.resize_conv5 = nn.utils.spectral_norm(nn.Conv3d(channels[4], channels[5], kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_pix_shuff5 = pixOp.ConvPixelShuffleLite3d(channels[4], channels[5], upscale_factor=scale_step, padding=1, activation=self.activation, norm='instance', spectral_norm=True)
        self.in6 = nn.InstanceNorm3d(channels[5], affine=True)

        self.conv7 = nn.utils.spectral_norm(nn.Conv3d(channels[5], 1, kernel_size=3, stride=1, padding=1))

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, noise: torch.Tensor) -> torch.Tensor:

        # 1024 x 1 x 1 x 1

        noise = noise.view(-1, self.noise, 1, 1, 1)
        h = self.tp_conv1(noise)
        h = self.activation(self.in1(h))

        # 512 x 4 x 4 x 4

        skip = F.interpolate(self.resize_conv1(h), scale_factor=2, mode='trilinear', align_corners=False)
        h = self.ssms_conv_pix_shuff1(h)
        h = self.activation(skip + self.in2(h))

        # 512 x 8 x 8 x 8
     
        skip = F.interpolate(self.resize_conv2(h), scale_factor=2, mode='trilinear', align_corners=False)
        h = self.conv_pix_shuff2(h)
        h = self.activation(skip + self.in3(h))
        
        # 256 x 16 x 16 x 16

        skip = F.interpolate(self.resize_conv3(h), scale_factor=2, mode='trilinear', align_corners=False)
        h = self.conv_pix_shuff3(h)
        h = self.activation(skip + self.in4(h))

        # 128 x 32 x 32 x 32

        skip = F.interpolate(self.resize_conv4(h), scale_factor=2, mode='trilinear', align_corners=False)
        h = self.ssms_conv_pix_shuff4(h)
        h = self.activation(skip + self.in5(h))

        # 64 x 64 x 64 x 64

        skip = F.interpolate(self.resize_conv5(h), scale_factor=2, mode='trilinear', align_corners=False)
        h = self.conv_pix_shuff5(h)
        h = self.activation(skip + self.in6(h))

        # 32 x 128 x 128 x 128

        h = self.conv7(h)
        h = self.tanh(h)

        # 1 x 128 x 128 x 128

        return h

class Discriminator(nn.Module):
    def __init__(self, channels:List[int]=[64, 256, 256, 512, 1024], out_class:int=1):
        super(Discriminator, self).__init__()

        self.channels = channels
        n_class = out_class
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.avgpool = nn.AvgPool3d(2)
        self.maxpool = nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.conv1 = nn.utils.spectral_norm(nn.Conv3d(1, channels[0], kernel_size=4, stride=2, padding=1))

        self.ssms_block2 = ssms.SSMultiscaleBlock3d(channels[0], channels[1], output_prop_2=0.3, output_prop_3=0.2, kernel_size_1=3, kernel_size_2=3, kernel_size_3=3, stride=2, dilation_1=(1, 1, 1), dilation_2=(1, 1, 1), dilation_3=(2, 2, 2), norm='None', activation=self.activation, bias=False, spectral_norm=True)
        self.bottleneck2 = nn.utils.spectral_norm(nn.Conv3d(channels[0] + channels[1], channels[1], kernel_size=1, stride=1, padding=0, bias=False))

        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False))
        self.bottleneck3 = nn.utils.spectral_norm(nn.Conv3d(channels[1] + channels[2], channels[2], kernel_size=1, stride=1, padding=0, bias=False))

        self.conv4 = nn.utils.spectral_norm(nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, bias=False))
        self.bottleneck4 = nn.utils.spectral_norm(nn.Conv3d(channels[2] + channels[3], channels[3], kernel_size=1, stride=1, padding=0, bias=False))

        self.ssms_block5 = ssms.SSMultiscaleBlock3d(channels[3], channels[4], output_prop_2=0.3, output_prop_3=0.2, kernel_size_1=3, kernel_size_2=3, kernel_size_3=3, stride=2, dilation_1=(1, 1, 1), dilation_2=(1, 1, 1), dilation_3=(1, 1, 1), norm='None', activation=self.activation, bias=False, spectral_norm=True)
        self.bottleneck5 = nn.utils.spectral_norm(nn.Conv3d(channels[3] + channels[4], channels[4], kernel_size=1, stride=1, padding=0, bias=False))

        self.conv6 = nn.utils.spectral_norm(nn.Conv3d(channels[4], n_class, kernel_size=1, stride=1, padding=0))

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 1 x 128 x 128 x 128

        h = self.activation(self.conv1(x))

        # 64 x 64 x 64 x 64

        h = torch.cat([self.maxpool(h), self.ssms_block2(h)], dim=1)
        h = self.activation(self.bottleneck2(h))

        # 256 x 32 x 32 x 32

        h = torch.cat([self.avgpool(h), self.activation(self.conv3(h))], dim=1)
        h = self.activation(self.bottleneck3(h))

        # 256 x 16 x 16 x 16

        h = torch.cat([self.avgpool(h), self.activation(self.conv4(h))], dim=1)
        h = self.activation(self.bottleneck4(h))
        
        # 512 x 8 x 8 x 8
        
        h = torch.cat([self.maxpool(h), self.ssms_block5(h)], dim=1)
        h = self.activation(self.bottleneck5(h))

        # 1024 x 4 x 4 x 4
        
        h = self.conv6(self.gap(h))

        # num_outputs x 1 x 1 x 1
        
        return h
