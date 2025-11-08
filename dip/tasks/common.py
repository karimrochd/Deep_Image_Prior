import torch
import torch.nn.functional as F
from ..utils.metrics import psnr

class BaseTask:
    name = "base"
    def __init__(self): pass
    def data_loss(self, y, obs, **kw): raise NotImplementedError
    def report(self, y, obs, gt=None):
        out = {'loss': self.data_loss(y, obs).item()}
        if gt is not None:
            out['psnr'] = psnr(y, gt).item()
        else:
            out['psnr'] = psnr(y, obs).item()  # heuristic
        return out

def make_task(name, **kw):
    name = name.lower()
    if name == 'denoise':
        return DenoiseTask()
    if name == 'superres':
        return SuperResTask(scale=kw.get('scale',4))
    if name == 'inpaint':
        return InpaintTask(mask=kw.get('mask'))
    raise ValueError(f'Unknown task: {name}')
