# LICENSE
# This file was extracted from
#   https://github.com/LS4GAN/uvcgan2
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE

import torch
from torch import nn

from .layers.transformer import calc_tokenized_size


class SequenceRandomMasking(nn.Module):

    def __init__(self, fraction = 0.4, seed = 0, **kwargs):
        super().__init__(**kwargs)
        self._fraction = fraction

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def forward(self, sequence):
        # sequence : (N, L, features)
        mask  = (
              torch.rand((*sequence.shape[:2], 1), generator = self._rng)
            > self._fraction
        )
        return mask.to(sequence.device) * sequence

class ImagePatchRandomMasking(nn.Module):

    def __init__(self, patch_size, fraction = 0.4, seed = 0, **kwargs):
        super().__init__(**kwargs)

        self._patch_size = patch_size
        self._fraction   = fraction

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def forward(self, image):
        # image : (N, C, H, W)
        N_h, N_w = calc_tokenized_size(image.shape[1:], self._patch_size)

        # mask : (N, 1, N_h, N_w)
        mask = (
              torch.rand((image.shape[0], 1, N_h, N_w), generator = self._rng)
            > self._fraction
        )

        # mask : (N, 1, N_h, N_w)
        #     -> (N, 1,   H,   W)
        mask = mask.repeat_interleave(self._patch_size[0], dim = 2)
        mask = mask.repeat_interleave(self._patch_size[1], dim = 3)

        return mask.to(image.device) * image