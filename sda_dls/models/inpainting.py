import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl

from fccgan.base.torch.weigth_init import init_weights

class InpaintingModel(pl.LightningModule):
        
    def __init__(
        self, 
        generator: nn.Module,
        masking_module: nn.Module,
        weight_init: callable,
        optimizer: callable,
        scheduler: callable,
        compile: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'masking_module'])
        
        self.gen = generator
        init_weights(self.gen, weight_init)
        self.masking = masking_module
        self.criterion = nn.L1Loss()
        
        self.imgs = {}
        
    def training_step(self, batch, batch_idx):
        self.imgs['real']= batch
        self.imgs['masked'] = self.masking(self.imgs['real'])
        self.imgs['rec'] = self.gen(self.imgs['masked'])
        
        loss = self.criterion(self.imgs['rec'], self.imgs['real'])
        
        self.log('loss', loss, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        real = batch
        masked = self.masking(real)
        rec = self.gen(masked)

        val_loss = self.criterion(rec, real)
        self.log('val_loss', val_loss, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        weights = self.gen.parameters()
        optimizer = self.hparams.optimizer(weights)
        scheduler = self.hparams.scheduler(optimizer)
        
        return [optimizer], [scheduler]

class TwoDomainInpaintingModel(pl.LightningModule):

    def __init__(
        self, 
        generator: nn.Module,
        masking_module: nn.Module,
        weight_init: callable,
        optimizer: callable,
        scheduler: callable,
        compile: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'masking_module'])
        self.automatic_optimization = False
        self.gen_A = generator
        self.gen_B = copy.deepcopy(generator)
        init_weights(self.gen_A, weight_init)
        init_weights(self.gen_B, weight_init)

        self.masking = masking_module
        self.criterion = nn.L1Loss()
        
        self.imgs = {}

    def setup(self, stage: str):
        if self.hparams.compile and stage == 'fit':
            torch.compile(self.gen_A)
            torch.compile(self.gen_B)

    def _single_generator_step(self, img, domain_A=True):
        gen = self.gen_A if domain_A else self.gen_B
        suffix = 'A' if domain_A else 'B'
        img_keys = ['real', 'masked', 'rec']
        img_keys = [f'{key}_{suffix}' for key in img_keys]

        self.imgs[img_keys[0]] = img
        self.imgs[img_keys[1]] = self.masking(self.imgs[img_keys[0]])
        self.imgs[img_keys[2]] = gen(self.imgs[img_keys[1]])

        loss = self.criterion(self.imgs[img_keys[2]], self.imgs[img_keys[0]])
        return loss
    
    def training_step(self, batch, batch_idx):
        real_A, real_B = batch
        optimizer_A, optimizer_B = self.optimizers()

        self.toggle_optimizer(optimizer_A)
        optimizer_A.zero_grad()
        loss_A = self._single_generator_step(real_A, domain_A=True)
        self.manual_backward(loss_A)
        optimizer_A.step()
        self.untoggle_optimizer(optimizer_A)

        self.toggle_optimizer(optimizer_B)
        optimizer_B.zero_grad()
        loss_B = self._single_generator_step(real_B, domain_A=False)
        self.manual_backward(loss_B)
        optimizer_B.step()
        self.untoggle_optimizer(optimizer_B)
        
        self.log('loss_A', loss_A, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('loss_B', loss_B, logger=True, on_step=True, on_epoch=False, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        scheduler_A, scheduler_B = self.lr_schedulers()
        scheduler_A.step()
        scheduler_B.step()
        
    def validation_step(self, batch, batch_idx):
        imgs, domain = batch
        domain = torch.flatten(domain)
        
        imgs_A = imgs[domain == 0]
        imgs_B = imgs[domain == 1]
        imgs = torch.cat([imgs_A, imgs_B], dim=0)

        val_loss = torch.tensor(0.0, device=self.device)
        if len(imgs_A) != 0:
            masked_A = self.masking(imgs_A)
            rec_A = self.gen_A(masked_A)
            val_loss += self.criterion(rec_A, imgs_A)

        if len(imgs_B) != 0:
            masked_B = self.masking(imgs_B)
            rec_B = self.gen_B(masked_B)
            val_loss += self.criterion(rec_B, imgs_B)
        
        self.log('val_loss', val_loss, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_A = self.hparams.optimizer(self.gen_A.parameters())
        optimizer_B = self.hparams.optimizer(self.gen_B.parameters())
        scheduler_A = self.hparams.scheduler(optimizer_A)
        scheduler_B = self.hparams.scheduler(optimizer_B)
        
        return [optimizer_A, optimizer_B], [scheduler_A, scheduler_B]


    