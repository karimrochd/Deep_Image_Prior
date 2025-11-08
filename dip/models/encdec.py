import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(out_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x): return self.net(x)

class EncoderDecoder(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, depth=5, base_ch=64):
        super().__init__()
        chs = [base_ch*(2**i) for i in range(depth)]
        self.enc = nn.ModuleList()
        cur = in_ch
        for c in chs:
            self.enc.append(Block(cur, c))
            cur = c
        self.dec = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.dec.append(Block(chs[i+1], chs[i]))
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, z):
        x = z
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            feats.append(x)
            if i != len(self.enc)-1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        for i, blk in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
            x = blk(x)
        return self.out(x)
