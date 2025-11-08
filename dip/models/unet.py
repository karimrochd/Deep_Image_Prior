import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, depth=5, base_ch=64, use_skips=True):
        super().__init__()
        self.depth = depth
        self.use_skips = use_skips
        chs = [base_ch*(2**i) for i in range(depth)]
        # encoder
        self.enc = nn.ModuleList()
        cur_in = in_ch
        for c in chs:
            self.enc.append(ConvBlock(cur_in, c))
            cur_in = c
        # decoder
        self.dec = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.dec.append(ConvBlock(chs[i+1] + (chs[i] if use_skips else 0), chs[i]))
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, z):
        feats = []
        x = z
        # encoder with downsampling
        for i, blk in enumerate(self.enc):
            x = blk(x)
            feats.append(x)
            if i != len(self.enc)-1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        # decoder with upsampling
        for i, blk in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
            if self.use_skips:
                x = torch.cat([x, feats[-(i+2)]], dim=1)  # skip from encoder
            x = blk(x)
        return self.out(x)
