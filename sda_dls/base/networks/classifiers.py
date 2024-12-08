from typing import Tuple, Optional
import torch
import torch.nn as nn

from torchvision.models.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

from sda_dls.base.torch.weigth_init import init_weights
from sda_dls.base.torch.funcs import module_weights_from_pl_ckpt


class ClassifierNetwork(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(ClassifierNetwork, self).__init__()
        self.num_classes = num_classes

        self.backbone = self._build_backbone()
        out_features = self._get_out_features()
        self.head = torch.nn.Linear(out_features, self.num_classes)

    def _build_backbone(self) -> torch.nn.Module:
        raise NotImplementedError

    def _get_out_features(self) -> int:
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            x = self.backbone(x)
        return x.shape[1]

    def forward(
        self, x, get_features: bool = False, detach_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        pred = self.head(features.detach() if detach_features else features)

        if get_features:
            return pred, features

        return pred

    def freeze_stages(self, num_stages: int) -> None:
        raise NotImplementedError

    def reset_head(self) -> None:
        in_features = self._get_out_features()
        self.head = torch.nn.Linear(in_features, self.num_classes)

    @property
    def is_pretrained(self) -> bool:
        raise NotImplementedError


class ConvNext(ClassifierNetwork):
    WEIGHTS_DICT = {
        "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_large": ConvNeXt_Large_Weights.IMAGENET1K_V1,
    }

    INIT_DICT = {
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
    }

    _NUM_STAGES_CONVNEXT = 4

    def _build_backbone(self):
        if self.network not in ConvNext.WEIGHTS_DICT:
            raise ValueError("network %s not in ConvNeXt family" % self.network)

        weights = ConvNext.WEIGHTS_DICT[self.network] if self.load_imagenet else None
        convnext = ConvNext.INIT_DICT[self.network](weights=weights)
        backbone = nn.Sequential(
            convnext.features,
            convnext.avgpool,
            convnext.classifier[0],
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        return backbone

    def __init__(
        self,
        num_classes: int,
        network: str = "convnext_tiny",
        load_imagenet: bool = False,
    ):
        self.network = network
        self.load_imagenet = load_imagenet
        super().__init__(num_classes)

    def freeze_stages(self, stages: int):
        if stages <= 0:
            return
        elif stages > ConvNext._NUM_STAGES_CONVNEXT:
            raise ValueError(
                "network only has %d stages, not %d" % ConvNext._NUM_STAGES_CONVNEXT,
                stages,
            )

        backbone = self.backbone[0]
        for i in range(2 * stages):
            backbone[i].eval()
            backbone[i].requires_grad_(False)

    @property
    def is_pretrained(self) -> bool:
        return self.load_imagenet


def init_classifier(
    classifier: ClassifierNetwork,
    weight_init: Optional[callable] = None,
    stages_to_freeze: int = 0,
    ckpt: Tuple[str, str] | None = None,
    load_backbone_only: bool = False,
) -> ClassifierNetwork:
    if ckpt is not None:
        if load_backbone_only:
            head = classifier.head
            classifier.head = None

        module_weights_from_pl_ckpt(classifier, ckpt)

        if load_backbone_only:
            classifier.head = head

    elif weight_init is not None:
        module = classifier.head if classifier.is_pretrained else classifier
        init_weights(module, weight_init)

    classifier.freeze_stages(stages_to_freeze)
    return classifier
