# LICENSE
# File extracted from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `sda_dls/base/LICENSE` for copyright

import functools
import torch.nn as nn

from sda_dls.base.torch.select import get_norm_layer_fn


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        image_shape,
        ndf=64,
        n_layers=3,
        norm="batch",
        max_mult=8,
        shrink_output=True,
        return_intermediate_activations=False,
    ):
        # pylint: disable=too-many-locals
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer_fn(norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(image_shape[0], ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_mult)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, max_mult)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*sequence)
        self.shrink_conv = None

        if shrink_output:
            self.shrink_conv = nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw
            )

        self._intermediate = return_intermediate_activations

    def forward(self, input):
        """Standard forward."""
        z = self.model(input)

        if self.shrink_conv is None:
            return z

        y = self.shrink_conv(z)

        if self._intermediate:
            return (y, z)

        return y
