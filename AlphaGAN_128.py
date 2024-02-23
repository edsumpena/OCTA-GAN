import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, noise:int=1024, channels:int=512):
        super(Generator, self).__init__()

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        self.noise = noise

        self.tp_conv1 = nn.utils.spectral_norm(nn.ConvTranspose3d(noise, channels, kernel_size=4, stride=1, padding=0, bias=False))
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn3 = nn.BatchNorm3d(channels // 2)

        self.conv4 = nn.utils.spectral_norm(nn.Conv3d(channels // 2, channels // 4, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn4 = nn.BatchNorm3d(channels // 4)

        self.conv5 = nn.utils.spectral_norm(nn.Conv3d(channels // 4, channels // 8, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn5 = nn.BatchNorm3d(channels // 8)

        self.conv6 = nn.utils.spectral_norm(nn.Conv3d(channels // 8, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        h = self.activation(self.bn1(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.conv2(h)
        h = self.activation(self.bn2(h))
     
        h = F.upsample(h,scale_factor = 2)
        h = self.conv3(h)
        h = self.activation(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.conv4(h)
        h = self.activation(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.conv5(h)
        h = self.activation(self.bn5(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.conv6(h)

        h = self.tanh(h)

        return h

class Discriminator(nn.Module):
    def __init__(self, channels:int=512, out_class:int=1):
        super(Discriminator, self).__init__()

        self.channels = channels
        n_class = out_class
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv3d(1, channels // 8, kernel_size=4, stride=2, padding=1))

        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(channels // 8, channels // 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn2 = nn.BatchNorm3d(channels // 4)

        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(channels // 4, channels // 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn3 = nn.BatchNorm3d(channels // 2)

        self.conv4 = nn.utils.spectral_norm(nn.Conv3d(channels // 2, channels, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn4 = nn.BatchNorm3d(channels)

        self.conv5 = nn.utils.spectral_norm(nn.Conv3d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn5 = nn.BatchNorm3d(channels)

        self.conv6 = nn.utils.spectral_norm(nn.Conv3d(channels, n_class, kernel_size=4, stride=1, padding=0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1 x 128 x 128 x 128

        h = self.leaky_relu(self.conv1(x))

        # 64 x 64 x 64 x 64

        h = self.leaky_relu(self.bn2(self.conv2(h)))

        # 128 x 32 x 32 x 32

        h = self.leaky_relu(self.bn3(self.conv3(h)))

        # 256 x 16 x 16 x 16

        h = self.leaky_relu(self.bn4(self.conv4(h)))

        # 512 x 8 x 8 x 8

        h = self.leaky_relu(self.bn5(self.conv5(h)))

        # 512 x 4 x 4 x 4

        h = self.conv6(h)

        # num_outputs x 1 x 1 x 1
        
        return h