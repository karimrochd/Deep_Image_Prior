import torch
import torch.nn.functional as F
from .common import BaseTask

class InpaintTask(BaseTask):
    name = 'inpaint'
    def __init__(self, mask=None):
        self.mask = mask  # [1,1,H,W] float 0/1
    def data_loss(self, y, obs, **kw):
        m = self.mask
        if m is None:
            raise ValueError('Mask must be provided for inpainting')
        # broadcast to channels
        while m.ndim < y.ndim:
            m = m.expand(-1,y.shape[1],-1,-1)
        return F.mse_loss(y*m, obs*m)
