import torch
import torch.nn as nn

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

class DecodeBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        resolution,
        is_last,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.is_last = is_last
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU(0.2)
        self.activatel = nn.Tanh()

    def forward(self,x):
        x = self.conv(x)
        if not self.is_last:
            x = self.bn2d(x)
            x = self.activate(x)
        else:
            x = self.activatel(x)
        return x