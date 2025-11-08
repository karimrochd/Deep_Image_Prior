from .unet import UNet
from .encdec import EncoderDecoder
from .resunet import ResUNet
from .convnext_lite import ConvNeXtLite

def make_model(name='unet', in_ch=32, out_ch=3, depth=5, features=64, skip=True):
    name = name.lower()
    if name == 'unet':
        return UNet(in_ch, out_ch, depth=depth, base_ch=features, use_skips=skip)
    if name == 'encdec':
        return EncoderDecoder(in_ch, out_ch, depth=depth, base_ch=features)
    if name == 'resunet':
        return ResUNet(in_ch, out_ch, depth=depth, base_ch=features, use_skips=skip)
    if name == 'convnext':
        return ConvNeXtLite(in_ch, out_ch, depth=depth, base_ch=features)
    raise ValueError(f'Unknown model: {name}')
