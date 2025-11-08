from PIL import Image
import numpy as np
import torch

def load_image(path, gray=False):
    img = Image.open(path).convert('L' if gray else 'RGB')
    arr = np.array(img).astype('float32') / 255.0
    if gray:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2,0,1)
    tensor = torch.from_numpy(arr)[None, ...]  # [1,C,H,W]
    return tensor

def save_image(tensor, path):
    t = tensor.detach().clamp(0,1).cpu().numpy()
    t = t[0]  # [C,H,W]
    if t.shape[0] == 1:
        arr = (t[0]*255.0 + 0.5).astype('uint8')
        img = Image.fromarray(arr, mode='L')
    else:
        arr = (t.transpose(1,2,0)*255.0 + 0.5).astype('uint8')
        img = Image.fromarray(arr, mode='RGB')
    img.save(path)
