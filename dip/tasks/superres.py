import torch
import torch.nn.functional as F
from .common import BaseTask
from ..operators.downsamplers import differentiable_downsample

class SuperResTask(BaseTask):
    name = 'superres'
    def __init__(self, scale=4):
        self.scale = scale
    def data_loss(self, y, obs, **kw):
        # y is HR; obs is LR
        y_ds = differentiable_downsample(y, self.scale)
        return F.mse_loss(y_ds, obs)
