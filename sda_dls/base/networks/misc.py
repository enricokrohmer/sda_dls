import torch.nn as nn

from sda_dls.base.torch.gradient_reversal import GradientReversalLayer
from sda_dls.base.torch.select import get_norm_layer_1D


class DiscriminatorDANN(nn.Module):
    def __init__(
        self,
        input_nc: int,
        ndf: int = 1024,
        norm: str = "batch",
        n_layers: int = 1,
        sigmoid: bool = False,
    ):
        super().__init__()

        norm_layer = get_norm_layer_1D(norm, ndf)
        module_list = [
            nn.Linear(input_nc, ndf, bias=False),
            nn.ReLU(),
            norm_layer,
        ]

        for _ in range(n_layers):
            module_list += [
                nn.Linear(ndf, ndf, bias=False),
                nn.ReLU(),
                norm_layer,
            ]

        module_list += [
            nn.Linear(ndf, 1),
        ]

        if sigmoid:
            module_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*module_list)

    def forward(self, features, lambda_p):
        rev_features = GradientReversalLayer.apply(features, lambda_p)
        output = self.net(rev_features)
        return output


class ProjectionNet(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ndf: int = 1024,
        n_layers: int = 1,
        norm: str = "batch",
        normalize_output: bool = False,
    ):
        super().__init__()

        self.normalize_output = normalize_output
        norm_layer = get_norm_layer_1D(norm, ndf)

        layers = [
            nn.Linear(input_nc, ndf, bias=False),
            nn.ReLU(),
            norm_layer,
        ]

        for _ in range(n_layers):
            layers += [
                nn.Linear(ndf, ndf, bias=False),
                nn.ReLU(),
                norm_layer,
            ]

        layers.append(nn.Linear(ndf, output_nc, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        if self.normalize_output:
            y = nn.functional.normalize(y)

        return y
