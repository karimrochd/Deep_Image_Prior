from PIL import Image
import numpy as np
import torch

def load_mask(path, size=None):
    m = Image.open(path).convert('L')
    if size is not None:
        m = m.resize(size[::-1], resample=Image.NEAREST)
    arr = (np.array(m) > 127).astype('float32')
    t = torch.from_numpy(arr)[None,None,...]  # [1,1,H,W]
    return t

def random_bernoulli_mask(shape, p_keep=0.5, seed=0):
    torch.manual_seed(seed)
    m = (torch.rand(shape) < p_keep).float()
    return m
