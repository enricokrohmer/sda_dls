from typing import Optional, Tuple
import itertools
from torch import nn
import torch
from torch._tensor import Tensor
from torchmetrics import MetricCollection
import copy

from fccgan.models.strategies.classifier_strategies import Strategy, AbstractClassifierStrategy
from fccgan.base.networks.classifiers import ClassifierNetwork, init_classifier
from fccgan.base.networks.generators import init_generator
from fccgan.base.torch.funcs import update_average_model, domain_wise_metrics
from fccgan.base.torch.losses import EOSLoss

def entropy(x: Tensor) -> Tensor:
    x = torch.softmax(x, dim=1)
    return -torch.sum(x * torch.log(x + 1e-8), dim=1)

class DualHeadStrategy(Strategy):
    
    def __init__(
        self,
        classifier: ClassifierNetwork,
        label_smoothing: float,
        weight_init: callable,
        nc_src: int,
        thresh: float,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
    ):
        super().__init__()
        self.net_C = init_classifier(
            classifier, 
            weight_init, 
            stages_to_freeze, 
            pretrained_path, True
        )
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        out_feat = self.net_C._get_out_features()
        self.head_A = torch.nn.Linear(out_feat, nc_src, bias=False)
        self.head_B = torch.nn.Linear(out_feat, self.net_C.num_classes - nc_src, bias=False)
                    
        self.val_metrics = val_metrics
        self.val_open_set = copy.deepcopy(val_metrics)
        self.test_metrics = test_metrics
        
        self.num_classes = self.net_C.num_classes
        self.nc_src = nc_src
        self.thresh = thresh
        
    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        
        _, self.features_A = self.net_C(imgs_A, get_features=True, detach_features=True)
        _, self.features_B = self.net_C(imgs_B, get_features=True, detach_features=True)
        self.preds_A = self.head_A(self.features_A)
        self.preds_B = self.head_B(self.features_B)
        self.labels_A = labels_A
        self.labels_B = labels_B - self.nc_src
        
    def _eval_task_loss(self):
        task_loss_A = self.criterion_task(self.preds_A, self.labels_A)
        task_loss_B = self.criterion_task(self.preds_B, self.labels_B)
        task_loss = task_loss_A + task_loss_B
        self.pl_module.log('task_loss_A', task_loss_A, on_step=True, on_epoch=False, sync_dist=True)
        self.pl_module.log('task_loss_B', task_loss_B, on_step=True, on_epoch=False, sync_dist=True)
        return task_loss
    
    def get_loss(self):
        loss = self._eval_task_loss()
        return loss
        
    def _forward_eval(self, batch):
        imgs, labels = batch
        
        is_src = labels < self.nc_src
        imgs_A = imgs[is_src]
        imgs_B = imgs[~is_src]
        labels_A = labels[is_src]
        labels_B = labels[~is_src]
        
        _, feat_A = self.net_C(imgs_A, True)
        _, feat_B = self.net_C(imgs_B, True)
        #Without open set classification
        if is_src.any():
            preds_A = torch.argmax(self.head_A(feat_A), 1)
        else:
            preds_A = torch.tensor([]).to(imgs_A.device)

        if not is_src.all():
            preds_B = torch.argmax(self.head_B(feat_B), 1) + self.nc_src
        else:
            preds_B = torch.tensor([]).to(imgs_B.device)
        
        preds = torch.cat((preds_A, preds_B), dim=0)
        labels_f = torch.cat((labels_A, labels_B), dim=0)
        
        #With open set classification
        _, feat = self.net_C(imgs, True)

        preds_oB = self.head_B(feat)
        ood_scores = entropy(preds_oB) > self.thresh

        if ood_scores.any():
            
            preds_oA = self.head_A(
                feat[ood_scores]
            )
            preds_oA = torch.argmax(preds_oA, dim=1)
            labels_oA = labels[ood_scores]
            
            preds_oB = preds_oB[~ood_scores]
            preds_oB = torch.argmax(preds_oB, dim=1) + self.nc_src
            labels_oB = labels[~ood_scores]
            
            preds_o = torch.cat((preds_oA, preds_oB), dim=0)
            labels_o = torch.cat((labels_oA, labels_oB), dim=0)
        else:
            preds_o = torch.argmax(preds_oB, dim=1) + self.nc_src
            labels_o = labels

        return labels_f, preds, labels_o, preds_o
    
    def update_metrics(self, outputs, val=True):
        labels, preds, labels_o, preds_o = outputs
        if val:
            self.val_metrics.update(labels, preds)
            self.val_open_set.update(labels_o, preds_o)
        else:
            self.test_metrics.update(labels_o, preds_o)
        
    def compute_metrics(self, val = True):
        if val:
            metric_dict = domain_wise_metrics(self.val_metrics.compute(), self.nc_src)
            metric_dict_o = domain_wise_metrics(self.val_open_set.compute(), self.nc_src)
            metric_dict_o = {
                'open_set_' + key: value for key, value in metric_dict_o.items()
            }
            
            self.pl_module.log_dict(metric_dict_o, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.val_metrics.reset()
            self.val_open_set.reset()
        else:
            metric_dict = domain_wise_metrics(self.test_metrics.compute(), self.nc_src)
            self.test_metrics.compute()
            self.test_metrics.reset()
            
        self.pl_module.log_dict(metric_dict, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
    def get_weights(self):
        return itertools.chain(
            self.net_C.parameters(),
            self.head_A.parameters(),
            self.head_B.parameters(),
        )
    
    def _update_ema(self):
        pass
    
class OodClassDualHeadStrategy(Strategy):
    
    def __init__(
        self,
        classifier: ClassifierNetwork,
        label_smoothing: float,
        weight_init: callable,
        nc_src: int,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
    ):
        super().__init__()
        self.net_C = init_classifier(
            classifier, 
            weight_init, 
            stages_to_freeze, 
            pretrained_path, True
        )
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        out_feat = self.net_C._get_out_features()
        self.head_A = torch.nn.Linear(out_feat, nc_src, bias=False)
        self.head_B = torch.nn.Linear(out_feat, self.net_C.num_classes - nc_src + 1, bias=False)
        
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        
        self.num_classes = self.net_C.num_classes
        self.nc_src = nc_src
        
    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        
        _, self.features_A = self.net_C(imgs_A, get_features=True, detach_features=True)
        _, self.features_B = self.net_C(imgs_B, get_features=True, detach_features=True)
        self.preds_A = self.head_A(self.features_A)
        self.preds_B = self.head_B(torch.cat((self.features_B, self.features_A), dim=0))  
        
        self.labels_A = labels_A
        
        labels_B = labels_B - self.nc_src 
        ood_labels = torch.tensor(self.num_classes - self.nc_src).repeat(
            self.preds_A.shape[0]).to(imgs_A.device)
        
        self.labels_B = torch.cat((labels_B, ood_labels))
        
    def _eval_task_loss(self):
        task_loss_A = self.criterion_task(self.preds_A, self.labels_A)
        task_loss_B = self.criterion_task(self.preds_B, self.labels_B)
        task_loss = task_loss_A + task_loss_B
        self.pl_module.log('task_loss_A', task_loss_A, on_step=True, on_epoch=False, sync_dist=True)
        self.pl_module.log('task_loss_B', task_loss_B, on_step=True, on_epoch=False, sync_dist=True)
        return task_loss
    
    def get_loss(self):
        loss = self._eval_task_loss()
        return loss
    
    def _forward_eval(self, batch):
        imgs, labels = batch

        #With open set classification
        _, feat = self.net_C(imgs, True)
        
        preds_oB = torch.argmax(self.head_B(feat), dim=1)
        is_ood = preds_oB == self.num_classes - self.nc_src
        
        if is_ood.any():
            preds_oA = self.head_A(
                feat[is_ood]
            )
            preds_oA = torch.argmax(preds_oA, dim=1)
            labels_oA = labels[is_ood]
            
            preds_oB = preds_oB[~is_ood] + self.nc_src
            labels_oB = labels[~is_ood]
            
            preds_o = torch.cat((preds_oA, preds_oB), dim=0)
            labels_o = torch.cat((labels_oA, labels_oB), dim=0)
        else:
            preds_o = preds_oB + self.nc_src
            labels_o = labels
            
        return labels_o, preds_o
    
    def update_metrics(self, outputs, val=True):
        labels, preds = outputs
        if val:
            self.val_metrics.update(labels, preds)
        else:
            self.test_metrics.update(labels, preds)
        
    def compute_metrics(self, val = True):
        if val:
            metric_dict = domain_wise_metrics(self.val_metrics.compute(), self.nc_src)
            
            self.pl_module.log_dict(metric_dict, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.val_metrics.reset()
        else:
            metric_dict = domain_wise_metrics(self.test_metrics.compute(), self.nc_src)
            self.test_metrics.compute()
            self.test_metrics.reset()
            
        self.pl_module.log_dict(metric_dict, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
    def get_weights(self):
        return itertools.chain(
            self.net_C.parameters(),
            self.head_A.parameters(),
            self.head_B.parameters(),
        )
    
    def _update_ema(self):
        pass
            
    
class EntropicDualHeadStrategy(DualHeadStrategy):
    
    def __init__(
        self,
        classifier: ClassifierNetwork,
        generator: Optional[nn.Module],
        transforms: Optional[nn.Module],
        label_smoothing: float,
        weight_init: callable,
        nc_src: int,
        thresh: float,
        lambda_eos: float,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        pretrained_path: Optional[Tuple[str, str]] = None,
        pretrained_path_G: Optional[Tuple[str, str]] = None,
        stages_to_freeze: int = 0,
    ):
        super().__init__(
            classifier,
            label_smoothing,
            weight_init,
            nc_src,
            thresh,
            val_metrics,
            test_metrics,
            pretrained_path,
            stages_to_freeze,
        )
        if generator:
            self.net_G = init_generator(generator, weight_init, pretrained_path_G)
            self.net_G.eval()
            self.net_G.requires_grad_(False)
        else:
            self.net_G = None
        
        self.transforms = transforms
        self.criterion_open = EOSLoss(self.num_classes - self.nc_src)
        self.lambda_eos = lambda_eos
        
    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        if self.net_G:
            imgs_A = self.net_G(imgs_A)
            
        imgs_A = self.transforms(imgs_A)
        imgs_B = self.transforms(imgs_B)
        
        super()._forward_train((imgs_A, imgs_B, labels_A, labels_B))
        self.open_preds = self.head_B(self.features_A)
        
    def _eval_open_loss(self):
        open_loss = self.criterion_open(self.open_preds)
        self.pl_module.log('open_loss', open_loss, on_step=True, on_epoch=False, sync_dist=True)
        return open_loss
    
    def get_loss(self):
        loss = super().get_loss()
        loss += self.lambda_eos * self._eval_open_loss()
        return loss
    
    
class TwinClassifierStrategy(AbstractClassifierStrategy):
    
    def __init__(
        self,
        classifier: ClassifierNetwork,
        classifier_B: ClassifierNetwork,
        label_smoothing: float,
        thresh: float,
        weight_init: callable,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        stages_to_freeze: int = 0,
        stages_to_freeze_B: int = 0,
        nc_src : Optional[int] = None,
    ):
        super().__init__(
            classifier=classifier,
            label_smoothing=label_smoothing,
            weight_init=weight_init,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            stages_to_freeze=stages_to_freeze,
            nc_src=nc_src,
        )
        self.net_Cb = init_classifier(
            classifier_B, weight_init, stages_to_freeze_B
        )
        self.thresh = thresh
        self.val_open_set = copy.deepcopy(self.val_metrics)
        
    def _forward_train(self, batch):
        imgs_A, imgs_B, labels_A, labels_B = batch
        
        self.preds_A = self.net_C(imgs_A)
        self.preds_B = self.net_Cb(imgs_B)
        self.labels_A = labels_A
        self.labels_B = labels_B - self.nc_src
        
    def _eval_task_loss(self):
        task_loss_A = self.criterion_task(self.preds_A, self.labels_A)
        self.pl_module.log('task_loss_A', task_loss_A, on_step=True, on_epoch=False, sync_dist=True)
        task_loss_B = self.criterion_task(self.preds_B, self.labels_B)
        self.pl_module.log('task_loss_B', task_loss_B, on_step=True, on_epoch=False, sync_dist=True)

        return task_loss_A + task_loss_B
    
    def get_loss(self):
        return self._eval_task_loss()
    
    def _forward_eval(self, batch):
        imgs, labels = batch
        
        is_src = labels < self.nc_src
        imgs_A = imgs[is_src]
        imgs_B = imgs[~is_src]
        labels_A = labels[is_src]
        labels_B = labels[~is_src]
        
        #Without open set classification
        if is_src.any():
            preds_A = torch.argmax(self.net_C(imgs_A), 1)
        else:
            preds_A = torch.tensor([]).to(imgs_A.device)

        if not is_src.all():
            preds_B = torch.argmax(self.net_Cb(imgs_B), 1) + self.nc_src
        else:
            preds_B = torch.tensor([]).to(imgs_B.device)
        
        preds = torch.cat((preds_A, preds_B), dim=0)
        labels = torch.cat((labels_A, labels_B), dim=0)
        
        #With open set classification
        preds_oB = self.net_Cb(
            torch.cat((imgs_A, imgs_B), dim=0)
        )
        ood_scores = entropy(preds_oB) > self.thresh

        if ood_scores.any():
            
            preds_oA = self.net_C(
                torch.cat((imgs_A, imgs_B), dim=0)[ood_scores]
            )
            preds_oA = torch.argmax(preds_oA, dim=1)
            labels_oA = labels[ood_scores]
            
            preds_oB = preds_oB[~ood_scores]
            preds_oB = torch.argmax(preds_oB, dim=1) + self.nc_src
            labels_oB = labels[~ood_scores]
            
            preds_o = torch.cat((preds_oA, preds_oB), dim=0)
            labels_o = torch.cat((labels_oA, labels_oB), dim=0)
        else:
            preds_o = torch.argmax(preds_oB, dim=1) + self.nc_src
            labels_o = labels

        return labels, preds, labels_o, preds_o

    def update_metrics(self, outputs, val=True):
        labels, preds, labels_o, preds_o = outputs
        if val:
            self.val_metrics.update(labels, preds)
            self.val_open_set.update(labels_o, preds_o)
        else:
            self.test_metrics.update(labels_o, preds_o)
        
    def compute_metrics(self, val = True):
        if val:
            metric_dict = domain_wise_metrics(self.val_metrics.compute(), self.nc_src)
            metric_dict_o = domain_wise_metrics(self.val_open_set.compute(), self.nc_src)
            metric_dict_o = {
                'open_set_' + key: value for key, value in metric_dict_o.items()
            }
            
            self.pl_module.log_dict(metric_dict_o, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.val_metrics.reset()
            self.val_open_set.reset()
        else:
            metric_dict = domain_wise_metrics(self.test_metrics.compute(), self.nc_src)
            self.test_metrics.compute()
            self.test_metrics.reset()
            
        self.pl_module.log_dict(metric_dict, logger=True, on_step=False, on_epoch=True, sync_dist=True)
             
    def get_weights(self):
        return itertools.chain(
            self.net_C.parameters(),
            self.net_Cb.parameters()
        )
    
    def _update_ema(self):
        pass