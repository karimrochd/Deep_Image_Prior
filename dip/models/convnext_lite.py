import torch
import torch.nn as nn
import torch.nn.functional as F

# A very small ConvNeXt-like block (depthwise + pointwise + GELU)
class CNBlock(nn.Module):
    def __init__(self, ch, expansion=2.0):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        hidden = int(ch*expansion)
        self.pw1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        res = x
        x = self.dw(x)
        x = self.pw2(self.act(self.pw1(x)))
        return res + self.gamma * x

class ConvNeXtLite(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, depth=4, base_ch=64):
        super().__init__()
        chs = [base_ch*(2**i) for i in range(depth)]
        self.stems = nn.ModuleList()
        self.blocks_enc = nn.ModuleList()
        cur = in_ch
        for c in chs:
            self.stems.append(nn.Conv2d(cur, c, 3, padding=1))
            self.blocks_enc.append(CNBlock(c))
            cur = c
        self.proj_down = nn.ModuleList([nn.Conv2d(chs[i], chs[i+1], 2, stride=2) for i in range(depth-1)])
        self.blocks_dec = nn.ModuleList([CNBlock(chs[i]) for i in range(depth-2, -1, -1)])
        self.proj_up = nn.ModuleList([nn.ConvTranspose2d(chs[i+1], chs[i], 2, stride=2) for i in range(depth-1)])
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, z):
        xs = []
        x = z
        for i in range(len(self.stems)):
            x = self.stems[i](x)
            x = self.blocks_enc[i](x)
            xs.append(x)
            if i != len(self.stems)-1:
                x = self.proj_down[i](x)
        for i in range(len(self.blocks_dec)):
            x = self.proj_up[i](x)
            x = x + xs[-(i+2)]
            x = self.blocks_dec[i](x)
        return self.out(x)
