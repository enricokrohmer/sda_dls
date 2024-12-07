from typing import Optional, Sequence
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import openloss

from sda_dls.base.networks.classifiers import ClassifierNetwork, init_classifier

class GANLoss(nn.Module):
    """ This class was extracted from
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        Please see `sda_dls/base/LICENSE` for copyright attribution and LICENSE
    """

    def __init__(
        self, gan_mode, target_real_label = 1.0, target_fake_label = 0.0
    ):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objective.
                Choices: vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image
            target_fake_label (bool) -- label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with
        BCEWithLogitsLoss.
        """
        super().__init__()

        # pylint: disable=not-callable
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

        elif gan_mode == 'softplus':
            self.loss = nn.Softplus()

        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) -- tpyically the prediction from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of
            the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) -- tpyically the prediction output from a
                discriminator
            target_is_real (bool) -- if the ground truth label is for real
                images or fake images

        Returns:
            the calculated loss.
        """

        if isinstance(prediction, (list, tuple)):
            result = sum(self.forward(x, target_is_real) for x in prediction)
            return result / len(prediction)

        if self.gan_mode == 'softplus':
            if target_is_real:
                return self.loss(prediction).mean()
            else:
                return self.loss(-prediction).mean()

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
    
    
class FeatureConsistencyLoss(nn.Module):
    
    def __init__(
        self,
        lambda_fc : float,
        network: ClassifierNetwork,
        pretrained_path: Optional[str] = None,
        margin : float = 1.0,
    ):
        super().__init__()
        if lambda_fc <= 0:
            raise ValueError(
                'lambda_fc must be greater than zero, got {}'.format(lambda_fc)
            )
        self.lambda_fc = lambda_fc
        
        self.network = init_classifier(classifier=network, ckpt=pretrained_path)
        self.network.eval()
        self.network.requires_grad_(False)
        
        if margin <= 0:
            raise ValueError(
                'margin must be greater than zero, got {}'.format(margin)
            )
        self.loss = nn.TripletMarginLoss(margin=margin)
        
        
    def forward(self, anchor, positive, negative):
        _, anchor_f = self.network(anchor, get_features=True)
        _, positive_f = self.network(positive, get_features=True)
        _, negative_f = self.network(negative, get_features=True)
        loss = self.lambda_fc * \
            self.loss(anchor_f, positive_f, negative_f)
        
        return loss
    

# LICENSE
# The remaining code was extracted from
#  https://github.com/thuml/Transfer-Learning-Library
# Please see `sda_dls/base/LICENSE` for copyright attribution and LICENSE
class GaussianKernel(nn.Module):

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))

def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:

    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class mk_MMD(nn.Module):

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(mk_MMD, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss