# LICENSE
# This file was extracted from
#  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `sda_dls/base/LICENSE` for copyright attribution and LICENSE

import torch
from torch.nn import init


def winit_func(m: torch.nn.Module, init_func: callable):
    classname = m.__class__.__name__

    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        init_func(m.weight)

        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net: torch.nn.Module, init_func: callable):
    if init_func is None:
        return

    net.apply(lambda m, init_func=init_func: winit_func(m, init_func))
