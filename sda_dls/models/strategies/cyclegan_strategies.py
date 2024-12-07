from typing import Optional, Tuple, Union
import itertools
import copy
from abc import abstractmethod

import torch
from torch import nn
from kornia.enhance import denormalize

from fccgan.base.networks.generators import init_generator
from fccgan.base.networks.classifiers import init_classifier
from fccgan.base.torch.losses import GANLoss, PixelConsistencyLoss, FeatureConsistencyLoss
from fccgan.base.torch.image_pool import ImagePool
from fccgan.base.torch.gradient_penalty import GradientPenalty
from fccgan.base.torch.weigth_init import init_weights
from fccgan.base.torch.funcs import update_average_model
from fccgan.base.torch.queue import FastQueue
from fccgan.base.torch.layers.batch_head import BatchHeadWrapper, queued_forward

from .classifier_strategies import Strategy

class AbstractCycleGanStrategy(Strategy):

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion_gan: GANLoss,
        lambda_cycle: float,
        lambda_idt: float,
        weight_init: callable,
        criterion_fc : Optional[FeatureConsistencyLoss] = None,
        pretrained_path_A: Optional[Tuple[str, str]] = None,
        pretrained_path_B: Optional[Tuple[str, str]] = None,
        avg_momentum: float = 0,
        AtoB: bool = True,
    ):  
        super().__init__()
        
        # Initializing the networks
        gen_B = copy.deepcopy(generator)
        
        self.gen_A = init_generator(generator, weight_init, pretrained_path_A)
        self.gen_B = init_generator(gen_B, weight_init, pretrained_path_B)
        
        self.avg_momentum = avg_momentum
        if not (0 <= avg_momentum <= 1):
            raise ValueError(
                f'avg_momentum must be in the interval [0, 1], got {avg_momentum}'
            )
        if avg_momentum > 0:
            self.avg_A = copy.deepcopy(self.gen_A)
            self.avg_A.eval()
            self.avg_A.requires_grad_(False)
            self.avg_B = copy.deepcopy(self.gen_B)
            self.avg_B.eval()
            self.avg_B.requires_grad_(False)
        
        self.disc_A = discriminator
        self.disc_B = copy.deepcopy(discriminator)
        init_weights(self.disc_A, weight_init)
        init_weights(self.disc_B, weight_init)

        #Initializing losses
        self.lambda_cycle = lambda_cycle
        self.lambda_idt = lambda_idt

        self.criterion_gan = criterion_gan
        self.criterion_cycle = nn.L1Loss()
        self.criterion_fc = criterion_fc
        if self.lambda_idt > 0.0:
            self.criterion_idt = nn.L1Loss()

        self.imgs = {}
        self.AtoB = AtoB

    def _forward_train(self, batch):
        self.imgs['real_A'], self.imgs['real_B'] = batch
        self.imgs['fake_B'] = self.gen_A(self.imgs['real_A'])
        self.imgs['fake_A'] = self.gen_B(self.imgs['real_B'])

        if self.lambda_idt > 0:
            self.imgs['idt_A'] = self.gen_B(self.imgs['real_A'])
            self.imgs['idt_B'] = self.gen_A(self.imgs['real_B'])

        if self.avg_momentum > 0:
            self.imgs['avg_B'] = self.avg_A(self.imgs['real_A'])
            self.imgs['avg_A'] = self.avg_B(self.imgs['real_B'])
            
    def _forward_predict(self, batch):
        imgs, paths = batch
        gen = self.gen_A if self.AtoB else self.gen_B
        
        return gen(imgs), paths
            
    @abstractmethod
    def backward_discriminators(self):
        pass

    def _eval_gan_loss(self):
        loss_B = self.criterion_gan(self.disc_A(self.imgs['fake_A']), True)
        loss_A = self.criterion_gan(self.disc_B(self.imgs['fake_B']), True)

        self.pl_module.log("loss_gen_A", loss_A, on_step=True, on_epoch=False, sync_dist=True)
        self.pl_module.log("loss_gen_B", loss_B, on_step=True, on_epoch=False, sync_dist=True)

        return loss_A + loss_B

    def _eval_cycle_loss(self):
        self.imgs["rec_A"] = self.gen_B(self.imgs["fake_B"])
        self.imgs["rec_B"] = self.gen_A(self.imgs["fake_A"])

        loss_cycle_A = self.criterion_cycle(self.imgs["rec_A"], self.imgs["real_A"])
        loss_cycle_B = self.criterion_cycle(self.imgs["rec_B"], self.imgs["real_B"])
        loss_cycle = self.lambda_cycle * (loss_cycle_A + loss_cycle_B)

        self.pl_module.log("loss_cycle", loss_cycle, on_step=True, on_epoch=False, sync_dist=True)
        return loss_cycle

    def _eval_idt_loss(self):
        loss_idt_A = self.criterion_idt(self.imgs["idt_A"], self.imgs["real_A"])
        loss_idt_B = self.criterion_idt(self.imgs["idt_B"], self.imgs["real_B"])
        loss_idt = self.lambda_idt * (loss_idt_A + loss_idt_B)

        self.pl_module.log("loss_idt", loss_idt, on_step=True, on_epoch=False, sync_dist=True)
        return loss_idt
    
    def _eval_fc_loss(self):
        loss_fc_A = self.criterion_fc(self.imgs["real_A"], self.imgs["fake_B"], self.imgs["real_B"])
        loss_fc_B = self.criterion_fc(self.imgs["real_B"], self.imgs["fake_A"], self.imgs["real_A"])
        loss_fc = loss_fc_A + loss_fc_B
        
        self.pl_module.log("loss_fc", loss_fc, on_step=True, on_epoch=False, sync_dist=True)
        return loss_fc

    def _accumulate_gan_losses(self):
        loss = self._eval_gan_loss()
        loss += self._eval_cycle_loss()
        if self.lambda_idt > 0:
            loss += self._eval_idt_loss()
        if self.criterion_fc is not None:
            loss += self._eval_fc_loss()

        return loss

    def backward_generators(self):
        loss = self._accumulate_gan_losses()
        self.pl_module.manual_backward(loss)

    def _update_ema(self):
        if self.avg_momentum > 0:
            update_average_model(self.avg_A, self.gen_A, self.avg_momentum)
            update_average_model(self.avg_B, self.gen_B, self.avg_momentum)

    @abstractmethod
    def _setup_buffers(self):
        pass

    def get_weights(self):
        return [
            itertools.chain(
                self.gen_A.parameters(),
                self.gen_B.parameters(),
            ),
            itertools.chain(
                self.disc_A.parameters(),
                self.disc_B.parameters(),
            ),
        ]

    def cleanup(self):
        del(self.disc_A)
        del(self.disc_B)

        if self.avg_momentum:
            del(self.gen_A)
            del(self.gen_B)


class CycleGanStrategy(AbstractCycleGanStrategy):

    def __init__(
        self, 
        generator: nn.Module,
        discriminator: nn.Module,
        criterion_gan: GANLoss,
        lambda_cycle: float,
        lambda_idt: float,
        weight_init: callable,
        queue_size: int,
        criterion_fc : Optional[FeatureConsistencyLoss] = None,
        pretrained_path_A: Optional[Tuple[str, str]] = None,
        pretrained_path_B: Optional[Tuple[str, str]] = None,
        avg_momentum: float = 0,
        AtoB: bool = True,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            criterion_gan=criterion_gan,
            lambda_cycle=lambda_cycle,
            lambda_idt=lambda_idt,
            weight_init=weight_init,
            pretrained_path_A=pretrained_path_A,
            pretrained_path_B=pretrained_path_B,
            avg_momentum=avg_momentum,
            AtoB=AtoB,
            criterion_fc=criterion_fc,
        )
        self.queue_size = queue_size

    def _setup_buffers(self):
        self.pred_a_pool = ImagePool(self.queue_size)
        self.pred_b_pool = ImagePool(self.queue_size)

    def _bwd_single_disc(self, disc, real, fake):
        with torch.no_grad():
            fake = fake.contiguous()
            
        pred_real = disc(real)
        loss_real = self.criterion_gan(pred_real, True)
        pred_fake = disc(fake)
        loss_fake = self.criterion_gan(pred_fake, False)
        loss = (loss_real + loss_fake) * 0.5

        self.pl_module.manual_backward(loss)
        return loss
    
    def backward_discriminators(self):
        fake_A = self.pred_a_pool.query(self.imgs['fake_A'])
        fake_B = self.pred_b_pool.query(self.imgs['fake_B'])

        loss_discB = self._bwd_single_disc(self.disc_B, self.imgs['real_B'], fake_B)
        self.pl_module.log('loss_discB', loss_discB, on_step=True, on_epoch=False, sync_dist=True)
        loss_discA = self._bwd_single_disc(self.disc_A, self.imgs['real_A'], fake_A)
        self.pl_module.log('loss_discA', loss_discA, on_step=True, on_epoch=False, sync_dist=True)

    def cleanup(self):
        super().cleanup()
        del(self.pred_a_pool)
        del(self.pred_b_pool)
        
class Uvcganv2Strategy(AbstractCycleGanStrategy):

    def __init__(
        self, 
        generator: nn.Module,
        discriminator: nn.Module,
        batch_head: nn.Module,
        lambda_cycle,
        lambda_idt,
        weight_init: callable,
        criterion_gan: GANLoss,
        queue_size: int,
        gradient_penalty: Optional[GradientPenalty] = None,
        criterion_pixel: Optional[PixelConsistencyLoss] = None,
        pretrained_path_A: Optional[Tuple[str, str]] = None,
        pretrained_path_B: Optional[Tuple[str, str]] = None,
        criterion_fc : Optional[FeatureConsistencyLoss] = None,
        avg_momentum: float = 0,
        AtoB: bool = True,
    ):
        super().__init__(
            generator=generator,
            discriminator=BatchHeadWrapper(discriminator, batch_head),
            criterion_gan=criterion_gan,
            lambda_cycle=lambda_cycle,
            lambda_idt=lambda_idt,
            weight_init=weight_init,
            pretrained_path_A=pretrained_path_A,
            pretrained_path_B=pretrained_path_B,
            avg_momentum=avg_momentum,
            criterion_fc=criterion_fc,
            AtoB=AtoB,
        )

        self.criterion_pixel = criterion_pixel
        self.gp = gradient_penalty

        self.queue_size = queue_size

    def _setup_buffers(self):
        self.queues = {
            name: FastQueue(self.queue_size, self.pl_module.device) 
                for name in ("real_A", "real_B", "fake_A", "fake_B")
        }

    def _bwd_single_disc(self, disc, real, fake, q_real, q_fake):
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                disc, fake, real,
                model_kwargs_fake={"extra_bodies": q_fake.query()},
                model_kwargs_real={"extra_bodies": q_real.query()}
            )
            self.pl_module.manual_backward(loss_gp)

        pred_real = queued_forward(disc, real, q_real, update_queue=True)
        loss_real = self.criterion_gan(pred_real, True)
        pred_fake = queued_forward(disc, fake, q_fake, update_queue=True)
        loss_fake = self.criterion_gan(pred_fake, False)
        loss = (loss_real + loss_fake) * 0.5

        return loss, loss_gp

    def backward_discriminators(self):
        loss_discB, loss_gp_B = \
            self._bwd_single_disc(
                self.disc_B, self.imgs['real_B'], self.imgs['fake_B'].detach(), 
                self.queues["real_B"], self.queues["fake_B"]
            )
        loss_discA, loss_gp_A = \
            self._bwd_single_disc(
                self.disc_A, self.imgs["real_A"], self.imgs["fake_A"].detach(),
                self.queues["real_A"], self.queues["fake_A"]
            )

        if loss_gp_A is not None:
            self.pl_module.log("loss_gp", loss_gp_A + loss_gp_B, on_step=True, on_epoch=False, sync_dist=True)

        self.pl_module.log("loss_discA", loss_discA, on_step=True, on_epoch=False, sync_dist=True)
        self.pl_module.log("loss_discB", loss_discB, on_step=True, on_epoch=False, sync_dist=True)

        loss = loss_discA + loss_discB
        self.pl_module.manual_backward(loss)

    def _eval_pixel_loss(self):
        if self.criterion_pixel is None:
            return 0.0
        loss_pixel_A = self.criterion_pixel(self.imgs["fake_A"], self.imgs["real_A"])
        loss_pixel_B = self.criterion_pixel(self.imgs["fake_B"], self.imgs["real_B"])
        loss_pixel = loss_pixel_A + loss_pixel_B

        self.pl_module.log("loss_pixel", loss_pixel, on_step=True, on_epoch=False, sync_dist=True)
        return loss_pixel

    def _accumulate_gan_losses(self):
        loss = super()._accumulate_gan_losses()
        loss += self._eval_pixel_loss()

        return loss

    def cleanup(self):
        super().cleanup()
        del(self.queues)
        
        
class CyCADAStrategy(CycleGanStrategy):

    def __init__(
        self, 
        generator: nn.Module,
        discriminator: nn.Module,
        classifier: nn.Module,
        criterion_gan: GANLoss,
        lambda_cycle: float,
        lambda_sem: float,
        queue_size: int,
        weight_init: callable,
        pretrained_path_C: Tuple[str, str],
        pretrained_path_A: Optional[Tuple[str, str]] = None,
        pretrained_path_B: Optional[Tuple[str, str]] = None,
        avg_momentum: float = 0,
        AtoB: bool = True,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            criterion_gan=criterion_gan,
            queue_size=queue_size,
            lambda_cycle=lambda_cycle,
            lambda_idt=0,
            weight_init=weight_init,
            pretrained_path_A=pretrained_path_A,
            pretrained_path_B=pretrained_path_B,
            avg_momentum=avg_momentum,
            AtoB=AtoB,
            criterion_fc=None,
        )
        
        self.sem_C = init_classifier(classifier, weight_init, ckpt=pretrained_path_C)
        self.sem_C.eval()
        self.sem_C.requires_grad_(False)
        
        if lambda_sem <= 0:
            raise ValueError(f'lambda_sem must be a positive float, got {lambda_sem}')
        
        self.lambda_sem = lambda_sem
        self.criterion_sem = nn.CrossEntropyLoss()
        
    def _forward_train(self, batch):
        real_A, real_B, labels_A, labels_B = batch
        super()._forward_train((real_A, real_B))
        
        preds_fake_B = self.sem_C(self.imgs['fake_B'])
        preds_fake_A = self.sem_C(self.imgs['fake_A'])
        self.preds = torch.cat((preds_fake_B, preds_fake_A))
        self.labels = torch.cat((labels_A, labels_B))

    def _eval_sem_loss(self):
        loss_sem = self.lambda_sem * self.criterion_sem(
            self.preds, self.labels
        )
        
        self.pl_module.log("loss_sem", loss_sem, on_step=True, on_epoch=False, sync_dist=True)
        return loss_sem
    
    def _accumulate_gan_losses(self):
        loss = super()._accumulate_gan_losses()
        loss += self._eval_sem_loss()
        return loss
    
    def cleanup(self):
        super().cleanup()
        del(self.sem_C)
        
        
        