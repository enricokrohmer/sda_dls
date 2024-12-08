from typing import Any, Optional
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .strategies.classifier_strategies import ClassifierStrategy
from .strategies.gan_aided_strategies import TwoStepStrategy
from .strategies.cyclegan_strategies import AbstractCycleGanStrategy


class Classifier(LightningModule):
    def __init__(
        self,
        strategy: ClassifierStrategy,
        optimizer: callable,
        scheduler: callable,
        compile: bool = False,
        avg_momentum: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["strategy"])

        self.strategy = strategy
        strategy.attach_pl_module(self)

    def setup(self, stage: str):
        if self.hparams.compile:
            self.strategy.compile()

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        self.strategy._forward_train(batch)
        return self.strategy.get_loss()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        self.strategy._update_ema()

    def validation_step(self, batch, *args, **kwargs) -> Any:
        return self.strategy._forward_eval(batch)

    def on_validation_batch_end(self, outputs, *args, **kwargs) -> None:
        self.strategy.update_metrics(outputs)

    def on_validation_epoch_end(self) -> None:
        self.strategy.compute_metrics(val=True)

    def test_step(self, batch, *args, **kwargs):
        return self.strategy._forward_eval(batch)

    def on_test_batch_end(self, outputs, *args, **kwargs):
        self.strategy.update_metrics(outputs, val=False)

    def on_test_epoch_end(self):
        self.strategy.compute_metrics(val=False)

    def configure_optimizers(self):
        weights = self.strategy.get_weights()
        optimizer = self.hparams.optimizer(params=weights)
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]


class TwoStepClassifier(LightningModule):
    def __init__(
        self,
        task_strategy: TwoStepStrategy,
        cgan_strategy: AbstractCycleGanStrategy,
        lr_C: float,
        lr_G: float,
        lr_D: float,
        epochs_gan: int,
        wd_C: float,
        wd_gan: float,
        optimizer: callable,
        scheduler_C: callable,
        scheduler_gan: callable,
        compile: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["task_strategy", "cgan_strategy"])
        self.automatic_optimization = False

        self.task_strategy = task_strategy
        self.cgan_strategy = cgan_strategy

        self.task_strategy.attach_pl_module(self)
        self.cgan_strategy.attach_pl_module(self)

    def setup(self, stage: str):
        if self.hparams.compile:
            self.task_strategy.compile()
            self.cgan_strategy.compile()

    def on_train_start(self) -> None:
        self.cgan_strategy._setup_buffers()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.epochs_gan:
            self.cgan_strategy.cleanup()
            self.task_strategy.attach_gan(self.cgan_strategy)

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        optimizer_C, optimizer_G, optimizer_D = self.optimizers()

        if self.current_epoch < self.hparams.epochs_gan:
            self.cgan_strategy._forward_train(batch)

            self.toggle_optimizer(optimizer_D)
            optimizer_D.zero_grad()
            self.cgan_strategy.backward_discriminators()
            optimizer_D.step()
            self.untoggle_optimizer(optimizer_D)

            self.toggle_optimizer(optimizer_G)
            optimizer_G.zero_grad()
            self.cgan_strategy.backward_generators()
            optimizer_G.step()
            self.untoggle_optimizer(optimizer_G)

        else:
            self.task_strategy._forward_train(batch)

            self.toggle_optimizer(optimizer_C)
            optimizer_C.zero_grad()
            loss = self.task_strategy.get_loss()
            self.manual_backward(loss)
            optimizer_C.step()
            self.untoggle_optimizer(optimizer_C)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.current_epoch < self.hparams.epochs_gan:
            self.cgan_strategy._update_ema()
        else:
            self.task_strategy._update_ema()

    def on_train_epoch_end(self) -> None:
        scheduler_C, scheduler_G, scheduler_D = self.lr_schedulers()
        if self.current_epoch < self.hparams.epochs_gan:
            scheduler_G.step()
            scheduler_D.step()
        else:
            scheduler_C.step()

    def validation_step(self, batch, *args, **kwargs):
        return self.task_strategy._forward_eval(batch)

    def on_validation_batch_end(self, outputs, *args, **kwargs) -> None:
        self.task_strategy.update_metrics(outputs)

    def on_validation_epoch_end(self):
        self.task_strategy.compute_metrics(val=True)

    def test_step(self, batch, *args, **kwargs):
        return self.task_strategy._forward_eval(batch)

    def on_test_batch_end(self, outputs, *args, **kwargs) -> None:
        self.task_strategy.update_metrics(outputs, val=False)

    def on_test_epoch_end(self):
        self.task_strategy.compute_metrics(val=False)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        return self.cgan_strategy._forward_predict(batch)

    def get_imgs(self):
        if self.current_epoch < self.hparams.epochs_gan:
            return self.cgan_strategy.imgs
        else:
            return dict()

    def configure_optimizers(self):
        weights_G, weights_D = self.cgan_strategy.get_weights()
        weights_C = self.task_strategy.get_weights()

        lr_C = self.hparams.lr_C
        lr_G = self.hparams.lr_G
        lr_D = self.hparams.lr_D

        optimizer_C = self.hparams.optimizer(
            params=weights_C, lr=lr_C, weight_decay=self.hparams.wd_C
        )
        optimizer_G = self.hparams.optimizer(
            params=weights_G, lr=lr_G, weight_decay=self.hparams.wd_gan
        )
        optimizer_D = self.hparams.optimizer(
            params=weights_D, lr=lr_D, weight_decay=self.hparams.wd_gan
        )

        scheduler_C = self.hparams.scheduler_C(optimizer_C)
        scheduler_G = self.hparams.scheduler_gan(optimizer_G)
        scheduler_D = self.hparams.scheduler_gan(optimizer_D)

        return [optimizer_C, optimizer_G, optimizer_D], [
            scheduler_C,
            scheduler_G,
            scheduler_D,
        ]

    @property
    def imgs(self):
        if self.current_epoch < self.hparams.epochs_gan:
            return self.cgan_strategy.imgs
        else:
            return dict()
