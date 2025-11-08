import torch
import torch.nn.functional as F
from .common import BaseTask

class DenoiseTask(BaseTask):
    name = 'denoise'
    def data_loss(self, y, obs, **kw):
        return F.mse_loss(y, obs)
