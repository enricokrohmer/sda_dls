from abc import abstractmethod
import itertools
from typing import Optional, Tuple

import torch
from torch import nn
from torch._tensor import Tensor
from torchmetrics import MetricCollection
from pytorch_metric_learning.losses import GenericPairLoss

from sda_dls.base.networks.classifiers import ClassifierNetwork
from sda_dls.base.networks.misc import DiscriminatorDANN
from sda_dls.base.torch.weigth_init import init_weights
from .classifier_strategies import AbstractClassifierStrategy
from .cyclegan_strategies import AbstractCycleGanStrategy


class TwoStepStrategy(AbstractClassifierStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        transforms: nn.Module,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0.0,
        load_backbone_only: bool = False,
    ):
        super().__init__(
            classifier=classifier,
            label_smoothing=label_smoothing,
            weight_init=weight_init,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            pretrained_path=pretrained_path,
            stages_to_freeze=stages_to_freeze,
            nc_src=nc_src,
            best_metric_key=best_metric_key,
            maximize_best_metric=maximize_best_metric,
            avg_momentum=avg_momentum,
            load_backbone_only=load_backbone_only,
        )
        self.transforms = transforms
        self.transforms.requires_grad_(False)

    @abstractmethod
    def attach_gan(self, gan: AbstractCycleGanStrategy):
        pass


class GanAidedClassifierStrategy(TwoStepStrategy):
    def attach_gan(self, gan: AbstractCycleGanStrategy):
        self.net_G = gan.avg_A if gan.avg_momentum > 0 else gan.gen_A
        self.net_G.eval()
        self.net_G.requires_grad_(False)

    def _forward_train(self, batch):
        imgs, self.labels, domain = batch
        domain = torch.flatten(domain)

        if len(imgs[domain == 0]) != 0:
            imgs[domain == 0] = self.net_G(imgs[domain == 0])

        imgs = self.transforms(imgs)
        self.preds = self.net_C(imgs)

    def get_loss(self) -> torch.Tensor:
        loss = self._eval_task_loss()
        return loss


class SupConDAStrategy(TwoStepStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        projection_net: nn.Module,
        transforms: nn.Module,
        criterion_contrastive: GenericPairLoss,
        lambda_contrastive: float,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        detach_head: bool = False,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        one_sided: bool = True,
        avg_momentum: float = 0.0,
        load_backbone_only: bool = False,
    ):
        super().__init__(
            classifier=classifier,
            transforms=transforms,
            label_smoothing=label_smoothing,
            weight_init=weight_init,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            pretrained_path=pretrained_path,
            stages_to_freeze=stages_to_freeze,
            nc_src=nc_src,
            best_metric_key=best_metric_key,
            maximize_best_metric=maximize_best_metric,
            avg_momentum=avg_momentum,
            load_backbone_only=load_backbone_only,
        )
        self.net_P = projection_net
        self.contrastive_loss = criterion_contrastive
        self.lambda_contrastive = lambda_contrastive
        self.detach_head = detach_head
        self.one_sided = one_sided

    def get_weights(self):
        return itertools.chain(
            self.net_C.parameters(),
            self.net_P.parameters(),
        )

    def attach_gan(self, gan: AbstractCycleGanStrategy):
        self.gen_A = gan.avg_A if gan.avg_momentum > 0 else gan.gen_A
        self.gen_A.eval()
        self.gen_A.requires_grad_(False)

        if not self.one_sided:
            self.gen_B = gan.avg_B if gan.avg_momentum > 0 else gan.gen_B
            self.gen_B.eval()
            self.gen_B.requires_grad_(False)

    def _prepare_input_two_sided(self, batch):
        real_A, real_B, labels_A, labels_B = batch
        fake_B = self.gen_A(real_A)
        fake_A = self.gen_B(real_B)

        imgs = torch.cat([real_A, real_B, fake_B, fake_A], dim=0)
        imgs = self.transforms(imgs)

        labels = torch.cat([labels_A, labels_B, labels_A, labels_B], dim=0)
        return imgs, labels

    def _prepare_input_one_sided(self, batch):
        real_A, real_B, labels_A, labels_B = batch
        fake_B = self.gen_A(real_A)

        imgs = torch.cat([real_A, real_B, fake_B], dim=0)
        imgs = self.transforms(imgs)

        labels = torch.cat([labels_A, labels_B, labels_A], dim=0)
        return imgs, labels

    def _forward_train(self, batch):
        if self.one_sided:
            imgs, self.labels = self._prepare_input_one_sided(batch)
        else:
            imgs, self.labels = self._prepare_input_two_sided(batch)

        self.preds, self.features = self.net_C(
            imgs, get_features=True, detach_features=self.detach_head
        )

    def _eval_contrastive_loss(self):
        projections = self.net_P(self.features)
        labels = self.labels

        if self.pl_module.trainer.world_size > 1:
            projections, labels = self.pl_module.all_gather(
                [projections, labels], sync_grads=True
            )
            projections = torch.flatten(projections, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)

        loss = self.lambda_contrastive * self.contrastive_loss(projections, labels)
        self.pl_module.log(
            "loss_contrastive", loss, on_step=True, on_epoch=False, sync_dist=True
        )

        return loss

    def get_loss(self) -> torch.Tensor:
        loss = self._eval_task_loss()
        loss += self._eval_contrastive_loss()

        return loss


class CycadaDANN(TwoStepStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        discriminator: DiscriminatorDANN,
        transforms: nn.Module,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        lambda_disc: float,
        acc_threshold: float,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0.0,
        load_backbone_only: bool = False,
    ):
        super().__init__(
            classifier=classifier,
            label_smoothing=label_smoothing,
            transforms=transforms,
            weight_init=weight_init,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            pretrained_path=pretrained_path,
            stages_to_freeze=stages_to_freeze,
            nc_src=nc_src,
            best_metric_key=best_metric_key,
            maximize_best_metric=maximize_best_metric,
            avg_momentum=avg_momentum,
            load_backbone_only=load_backbone_only,
        )
        self.disc = discriminator
        init_weights(self.disc, weight_init)

        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.lambda_disc = lambda_disc
        self.acc_threshold = acc_threshold

        self.current_acc = torch.tensor(1.0)

    def attach_gan(self, gan: AbstractCycleGanStrategy):
        self.net_G = gan.avg_A if gan.avg_momentum > 0 else gan.gen_A
        self.net_G.eval()
        self.net_G.requires_grad_(False)

    def _update_acc(self, preds, labels):
        preds = torch.sigmoid(preds)
        if self.pl_module.trainer.world_size > 1:
            preds, labels = self.pl_module.all_gather([preds, labels], sync_grads=True)
            preds = torch.flatten(preds, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)

        self.current_acc = ((preds > 0.5) == labels).float().mean()
        self.pl_module.log(
            "acc_disc", self.current_acc, on_step=True, on_epoch=False, sync_dist=True
        )

    def _forward_train(self, batch):
        real_A, real_B, labels_A, labels_B = batch

        real_B = self.transforms(real_B)
        fake_B = self.transforms(self.net_G(real_A))

        preds_rB, features_rB = self.net_C(real_B, get_features=True)
        preds_fB, features_fB = self.net_C(fake_B, get_features=True)

        self.preds = torch.cat([preds_rB, preds_fB], dim=0)
        self.labels = torch.cat([labels_B, labels_A], dim=0)

        lambda_ = (
            self.lambda_disc
            if self.current_acc > self.acc_threshold
            else torch.tensor(0.0)
        )
        pred_disc_A = self.disc(features_rB, lambda_)
        pred_disc_B = self.disc(features_fB, lambda_)
        self.pred_disc = torch.cat([pred_disc_A, pred_disc_B], dim=0)
        self.domain = torch.cat(
            [torch.zeros_like(pred_disc_A), torch.ones_like(pred_disc_B)], dim=0
        )

    def _eval_disc_loss(self):
        loss = self.criterion_disc(self.pred_disc, self.domain)
        self._update_acc(self.pred_disc, self.domain)

        self.pl_module.log(
            "loss_disc", loss, on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def get_loss(self) -> Tensor:
        loss = self._eval_task_loss()
        loss += self._eval_disc_loss()

        return loss

    def get_weights(self):
        return itertools.chain(
            self.net_C.parameters(),
            self.disc.parameters(),
        )
