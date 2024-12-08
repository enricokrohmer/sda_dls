from typing import Any
import os
import torch
import torchvision
from lightning.pytorch import LightningModule, Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class ImageLoggingCallback(Callback):
    def __init__(self, print_freq: int, deactivation_epoch: int = 0):
        super().__init__()
        self.print_freq = print_freq
        self.deactivation_epoch = deactivation_epoch

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ) -> None:
        if (
            self.deactivation_epoch is not None
            and trainer.current_epoch >= self.deactivation_epoch
        ):
            return

        if not trainer.is_global_zero:
            return

        if trainer.global_step % self.print_freq == 0:
            img_dict = pl_module.imgs
            captions = list(img_dict.keys())
            imgs = [img_dict[k][0] for k in captions]  # Only one image per batch
            pl_module.logger.log_image(
                "samples", images=imgs, caption=captions, step=trainer.global_step
            )


class ImageStoringCallback(Callback):
    def __init__(self, target_root, mean, std):
        self.target_root = target_root
        self.mean = mean
        self.std = std

    def _renormalize(self, tensor):
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor = tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

        return tensor

    def _get_class_and_name(self, path):
        normalized = os.path.normpath(path)
        parts = normalized.split(os.sep)
        class_name = parts[-2]
        name = parts[-1]

        return class_name, name

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        *args,
        **kwargs,
    ) -> None:
        images, paths = outputs
        for image, path in zip(images, paths):
            image = self._renormalize(image)
            class_name, name = self._get_class_and_name(path)

            target_dir = os.path.join(
                self.target_root, str(trainer.current_epoch), class_name
            )
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            torchvision.utils.save_image(image, os.path.join(target_dir, name))
