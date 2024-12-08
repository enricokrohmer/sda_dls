# type: ignore

from typing import Optional, Tuple
import itertools
import copy
from abc import ABC, abstractmethod
from lightning.pytorch import LightningModule
from torch import nn
import torch
from torch._tensor import Tensor
from torchmetrics import MetricCollection, MaxMetric, MinMetric

from sda_dls.base.torch.losses import mk_MMD
from sda_dls.base.networks.classifiers import init_classifier, ClassifierNetwork
from sda_dls.base.networks.misc import DiscriminatorDANN
from sda_dls.base.torch.funcs import domain_wise_metrics, update_average_model
from sda_dls.base.torch.weigth_init import init_weights


class Strategy(ABC):
    def __init__(self):
        self.pl_module = None

    def attach_pl_module(self, module: LightningModule):
        module_dict = {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, nn.Module) and not isinstance(v, LightningModule)
        }
        for name, m in module_dict.items():
            module.add_module(name, m)

        self.pl_module = module

    def compile(self):
        module_list = [v for v in self.__dict__.values() if isinstance(v, nn.Module)]
        for m in module_list:
            torch.compile(m)

    @abstractmethod
    def get_weights(self):
        pass


class AbstractClassifierStrategy(Strategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0,
    ):
        super().__init__()
        self.net_C = init_classifier(
            classifier,
            weight_init,
            stages_to_freeze,
            pretrained_path,
        )
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.avg_momentum = avg_momentum
        if not (0 <= self.avg_momentum <= 1):
            raise ValueError(f"avg_momentum must be in [0, 1], got {self.avg_momentum}")

        if self.avg_momentum > 0:
            self.avg_C = copy.deepcopy(self.net_C)
            self.avg_C.eval()
            self.avg_C.requires_grad_(False)

        # Init metrics:
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.nc_src = nc_src
        self.best_metric_key = best_metric_key
        if best_metric_key is not None:
            best_metric_cls = MaxMetric if maximize_best_metric else MinMetric
            self.best_metric = best_metric_cls()

    @abstractmethod
    def _forward_train(self, batch):
        pass

    def _eval_task_loss(self):
        task_loss = self.criterion_task(self.preds, self.labels)
        self.pl_module.log(
            "task_loss", task_loss, on_step=True, on_epoch=False, sync_dist=True
        )

        return task_loss

    @abstractmethod
    def get_loss(self) -> torch.Tensor:
        pass

    def _forward_eval(self, batch):
        img, label = batch
        net = self.avg_C if self.avg_momentum > 0 else self.net_C
        pred = net(img)

        return label, pred

    def update_metrics(self, outputs, val=True):
        labels, preds = outputs
        metrics = self.val_metrics if val else self.test_metrics
        metrics.update(preds, labels)

    def _update_best_metric(self, metric_dict):
        current_metric = metric_dict[self.best_metric_key]
        self.best_metric.update(current_metric)
        metric_dict["best_val_metric"] = self.best_metric.compute()

        return metric_dict

    def compute_metrics(self, val=True):
        metrics = self.val_metrics if val else self.test_metrics
        metric_dict = metrics.compute()

        if self.nc_src:
            metric_dict = domain_wise_metrics(metric_dict, self.nc_src)

        if self.best_metric_key is not None and val:
            metric_dict = self._update_best_metric(metric_dict)

        self.pl_module.log_dict(
            metric_dict, logger=True, on_step=False, on_epoch=True, sync_dist=True
        )
        metrics.reset()

    def get_weights(self):
        return self.net_C.parameters()

    def _update_ema(self):
        if self.avg_momentum > 0:
            update_average_model(self.avg_C, self.net_C, self.avg_momentum)


class ClassifierStrategy(AbstractClassifierStrategy):
    def _forward_train(self, batch):
        img, self.labels = batch
        self.preds, self.features = self.net_C(img, get_features=True)

    def get_loss(self) -> torch.Tensor:
        loss = self._eval_task_loss()
        return loss


class TwoDomainClassifierStrategy(AbstractClassifierStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0,
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
        )

    def _forward_train(self, batch):
        imgs_A, imgs_B, self.labels_A, self.labels_B = batch
        self.preds_A, features_A = self.net_C(imgs_A, get_features=True)
        self.preds_B, features_B = self.net_C(imgs_B, get_features=True)
        self.preds = torch.cat((self.preds_A, self.preds_B), dim=0)
        self.labels = torch.cat((self.labels_A, self.labels_B), dim=0)
        self.features = torch.cat((features_A, features_B), dim=0)

    def get_loss(self) -> Tensor:
        loss = self._eval_task_loss()
        if self.logit_loss:
            loss += self._eval_logit_loss()
        return loss


class DANNStrategy(AbstractClassifierStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        discriminator: DiscriminatorDANN,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        gamma: float = 10,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0,
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
        )
        self.disc = discriminator
        init_weights(self.disc, weight_init)
        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.gamma = torch.tensor(gamma)

    def _get_lambda(self):
        max_steps = self.pl_module.trainer.estimated_stepping_batches
        current_step = self.pl_module.trainer.global_step

        p = torch.tensor(current_step / max_steps)
        return (1 - torch.exp(-self.gamma * p)) / (1 + torch.exp(-self.gamma * p))

    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        preds_A, features_A = self.net_C(imgs_A, get_features=True)
        preds_B, features_B = self.net_C(imgs_B, get_features=True)
        self.preds = torch.cat((preds_A, preds_B), dim=0)
        self.labels = torch.cat((labels_A, labels_B), dim=0)

        lambda_p = self._get_lambda() * 0.1
        disc_A = self.disc(features_A, lambda_p)
        disc_B = self.disc(features_B, lambda_p)
        self.disc_preds = torch.cat((disc_A, disc_B), dim=0)
        self.domain = torch.cat(
            (torch.zeros_like(disc_A), torch.ones_like(disc_B)), dim=0
        )

    def _eval_disc_loss(self):
        disc_loss = self.criterion_disc(self.disc_preds, self.domain)
        self.pl_module.log(
            "disc_loss", disc_loss, on_step=True, on_epoch=False, sync_dist=True
        )
        return disc_loss

    def get_loss(self) -> Tensor:
        task_loss = self._eval_task_loss()
        disc_loss = self._eval_disc_loss()
        loss = task_loss + disc_loss
        return loss

    def get_weights(self):
        return itertools.chain(self.net_C.parameters(), self.disc.parameters())


class DANStrategy(AbstractClassifierStrategy):
    def __init__(
        self,
        classifier: ClassifierNetwork,
        label_smoothing: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        lambda_mmd: float,
        criterion_mmd: mk_MMD,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
        nc_src: Optional[int] = None,
        best_metric_key: Optional[str] = None,
        maximize_best_metric: Optional[bool] = True,
        avg_momentum: float = 0,
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
        )
        self.criterion_mmd = criterion_mmd
        self.lambda_mmd = lambda_mmd

    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        self.preds_A, self.features_A = self.net_C(imgs_A, get_features=True)
        self.preds_B, self.features_B = self.net_C(imgs_B, get_features=True)

        self.preds = torch.cat((self.preds_A, self.preds_B), dim=0)
        self.labels = torch.cat((labels_A, labels_B), dim=0)

    def _eval_mmd_loss(self):
        if self.pl_module.trainer.world_size > 1:
            (
                features_A,
                features_B,
            ) = self.pl_module.all_gather(
                [self.features_A, self.features_B],
                sync_grads=True,
            )
            features_A = torch.flatten(features_A, start_dim=0, end_dim=1)
            features_B = torch.flatten(features_B, start_dim=0, end_dim=1)
        else:
            features_A = self.features_A
            features_B = self.features_B

        loss = self.lambda_mmd * self.criterion_mmd(features_A, features_B)

        self.pl_module.log(
            "mmd_loss", loss, on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def get_loss(self) -> Tensor:
        loss = self._eval_task_loss()
        loss += self._eval_mmd_loss()

        return loss
