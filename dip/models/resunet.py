import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = conv3x3(ch, ch)
        self.conv2 = conv3x3(ch, ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + res)

class Stem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x): return self.act(self.conv(x))

class ResUNet(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, depth=5, base_ch=64, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        chs = [base_ch*(2**i) for i in range(depth)]
        self.enc_stem = Stem(in_ch, chs[0])
        self.enc_blocks = nn.ModuleList([ResBlock(chs[0])])
        self.downs = nn.ModuleList()
        for i in range(depth-1):
            self.downs.append(nn.Conv2d(chs[i], chs[i+1], 3, stride=2, padding=1))
            self.enc_blocks.append(ResBlock(chs[i+1]))
        # decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.ups.append(nn.ConvTranspose2d(chs[i+1], chs[i], 4, stride=2, padding=1))
            in_ch_dec = chs[i]*(2 if use_skips else 1)
            self.dec_blocks.append(ResBlock(in_ch_dec))
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, z):
        x = self.enc_stem(z)
        feats = [x]
        for i in range(len(self.downs)):
            x = self.downs[i](x)
            x = self.enc_blocks[i+1](x)
            feats.append(x)
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            if self.use_skips:
                x = torch.cat([x, feats[-(i+2)]], dim=1)
            x = self.dec_blocks[i](x)
        return self.out(x)
