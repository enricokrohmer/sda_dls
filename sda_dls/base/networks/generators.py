from typing import Tuple
import functools
import numpy as np
import torch
import torch.nn as nn

from fccgan.base.torch.layers.residual_block import ResnetBlock
from fccgan.base.torch.layers.skip_connection_block import UnetSkipConnectionBlock
from fccgan.base.torch.layers.transformer import ExtendedPixelwiseViT
from fccgan.base.torch.layers.modnet import ModNet
from fccgan.base.torch.layers.transformer import PixelwiseViT
from fccgan.base.torch.layers.unet import UNet
from fccgan.base.torch.select import get_norm_layer_fn
from fccgan.base.torch.select import get_activ_layer
from fccgan.base.torch.weigth_init import init_weights
from fccgan.base.torch.funcs import module_weights_from_pl_ckpt
from fccgan.base.torch.layers.transformer import (
    calc_tokenized_size,
    ViTInput,
    TransformerEncoder,
    img_to_tokens,
    img_from_tokens,
)


class ResnetGenerator(nn.Module):
    """# LICENSE
        # This code was extracted from
        #  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        # Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf=64,
        norm="instance",
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()

        norm_layer = get_norm_layer_fn(norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetGenerator(nn.Module):
    """# LICENSE
# This file was extracted from
#  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE
"""

    def __init__(
        self, input_nc, output_nc, num_downs, ngf=64, norm="instance", use_dropout=False
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        norm_layer = get_norm_layer_fn(norm)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ViTGenerator(nn.Module):
    # LICENSE
# This file was extracted from
#   https://github.com/LS4GAN/uvcgan2
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE
    def __init__(
        self,
        features,
        n_heads,
        n_blocks,
        ffn_features,
        embed_features,
        activ,
        norm,
        input_shape,
        output_shape,
        token_size,
        rescale=False,
        rezero=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert input_shape == output_shape
        image_shape = input_shape

        self.image_shape = image_shape
        self.token_size = token_size
        self.token_shape = (image_shape[0], *token_size)
        self.token_features = np.prod([image_shape[0], *token_size])
        self.N_h, self.N_w = calc_tokenized_size(image_shape, token_size)
        self.rescale = rescale

        self.gan_input = ViTInput(
            self.token_features, embed_features, features, self.N_h, self.N_w
        )

        self.trans = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.gan_output = nn.Linear(features, self.token_features)

    # pylint: disable=no-self-use
    def calc_scale(self, x):
        # x : (N, C, H, W)
        return x.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8

    def forward(self, x):
        # x : (N, C, H, W)
        if self.rescale:
            scale = self.calc_scale(x)
            x = x / scale

        # itokens : (N, N_h, N_w, C, H_c, W_c)
        itokens = img_to_tokens(x, self.token_shape[1:])

        # itokens : (N, N_h,  N_w, C,  H_c,  W_c)
        #        -> (N, N_h * N_w, C * H_c * W_c)
        #         = (N, L,         in_features)
        itokens = itokens.reshape((itokens.shape[0], self.N_h * self.N_w, -1))

        # y : (N, L, features)
        y = self.gan_input(itokens)
        y = self.trans(y)

        # otokens : (N, L, in_features)
        otokens = self.gan_output(y)

        # otokens : (N, L, in_features)
        #        -> (N, N_h, N_w, C, H_c, W_c)
        otokens = otokens.reshape(
            (otokens.shape[0], self.N_h, self.N_w, *self.token_shape)
        )

        result = img_from_tokens(otokens)
        if self.rescale:
            result = result * scale

        return result


class ViTUNetGenerator(nn.Module):
    # LICENSE
# This file was extracted from
#   https://github.com/LS4GAN/uvcgan2
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE
    def __init__(
        self,
        features,
        n_heads,
        n_blocks,
        ffn_features,
        embed_features,
        activ,
        norm,
        image_shape,
        unet_features_list,
        unet_activ,
        unet_norm,
        unet_downsample="conv",
        unet_upsample="upsample-conv",
        unet_rezero=False,
        rezero=True,
        activ_output=None,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.net = UNet(
            unet_features_list,
            unet_activ,
            unet_norm,
            image_shape,
            unet_downsample,
            unet_upsample,
            unet_rezero,
        )

        bottleneck = PixelwiseViT(
            features,
            n_heads,
            n_blocks,
            ffn_features,
            embed_features,
            activ,
            norm,
            image_shape=self.net.get_inner_shape(),
            rezero=rezero,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)
        return self.output(result)


class ViTModNetGenerator(nn.Module):
    # LICENSE
# This file was extracted from
#   https://github.com/LS4GAN/uvcgan2
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE
    def __init__(
        self,
        features,
        n_heads,
        n_blocks,
        ffn_features,
        embed_features,
        activ,
        norm,
        image_shape,
        modnet_features_list,
        modnet_activ,
        modnet_norm=None,
        modnet_downsample="conv",
        modnet_upsample="upsample-conv",
        modnet_rezero=False,
        modnet_demod=True,
        rezero=True,
        activ_output=None,
        style_rezero=True,
        style_bias=True,
        n_ext=1,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.image_shape = image_shape

        mod_features = features * n_ext

        self.net = ModNet(
            modnet_features_list,
            modnet_activ,
            modnet_norm,
            image_shape,
            modnet_downsample,
            modnet_upsample,
            mod_features,
            modnet_rezero,
            modnet_demod,
            style_rezero,
            style_bias,
            return_mod=False,
        )

        bottleneck = ExtendedPixelwiseViT(
            features,
            n_heads,
            n_blocks,
            ffn_features,
            embed_features,
            activ,
            norm,
            image_shape=self.net.get_inner_shape(),
            rezero=rezero,
            n_ext=n_ext,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)
        return self.output(result)


def init_generator(
    generator: torch.nn.Module,
    weight_init: callable,
    pretrained_path: Tuple[str, str] = None,
):
    if pretrained_path is not None:
        module_weights_from_pl_ckpt(generator, pretrained_path)
    else:
        init_weights(generator, weight_init)

    return generator
