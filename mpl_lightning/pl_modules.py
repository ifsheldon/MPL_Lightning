from typing import Callable

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch import nn
from .aux_modules import SmoothCrossEntropy


class LightningMPL(pl.LightningModule):
    def __init__(self,
                 teacher_lr=0.01,
                 student_lr=0.01,
                 weight_decay=0.0,
                 enable_nesterov=True,
                 momentum=0.9,
                 temperature=1.0,
                 threshold=0.95,
                 lambda_u=1.0,
                 uda_steps=1,
                 label_smoothing=0,
                 ):
        """
        Init MPL
        :param teacher_lr: teacher training learning rate
        :param student_lr: student training learning rate
        :param weight_decay: weight decay
        :param enable_nesterov: use nesterov
        :param momentum: SGD Momentum
        :param temperature: pseudo label temperature
        :param threshold: pseudo label threshold
        :param lambda_u: coefficient of unlabeled loss
        :param uda_steps: warmup steps of lambda-u
        :param label_smoothing: label smoothing alpha
        """
        super(LightningMPL, self).__init__()
        self.save_hyperparameters()
        self.teacher = None  # FIXME
        self.student = None  # FIXME
        self.criterion = self.create_loss_fn()
        # activate manual optimization
        self.automatic_optimization = False

    def create_loss_fn(self):
        if self.hparams.label_smoothing > 0:
            criterion = SmoothCrossEntropy(alpha=self.hparams.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self):
        no_decay_parameter_subnames = ["bn"]
        teacher_parameters = [
            {'params': [parameter for name, parameter in self.teacher.named_parameters()
                        if not any(nd in name for nd in no_decay_parameter_subnames)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [parameter for name, parameter in self.teacher.named_parameters()
                        if any(nd in name for nd in no_decay_parameter_subnames)],
             'weight_decay': 0.0}
        ]
        student_parameters = [
            {'params': [parameter for name, parameter in self.student.named_parameters()
                        if not any(nd in name for nd in no_decay_parameter_subnames)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [parameter for name, parameter in self.student.named_parameters()
                        if any(nd in name for nd in no_decay_parameter_subnames)],
             'weight_decay': 0.0}
        ]

        t_optimizer = optim.SGD(teacher_parameters,
                                lr=self.hparams.teacher_lr,
                                momentum=self.hparams.momentum,
                                nesterov=self.hparams.enable_nesterov)
        s_optimizer = optim.SGD(student_parameters,
                                lr=self.hparams.student_lr,
                                momentum=self.hparams.momentum,
                                nesterov=self.hparams.enable_nesterov)
        return [t_optimizer, s_optimizer]

    def forward(self, image_batch):
        return self.student(image_batch)

    def training_step(self, batch, batch_idx):
        opt_teacher, opt_student = self.optimizers()
        images_labeled, targets, images_unlabeled_weak_aug, images_unlabeled_strong_aug = batch
        labeled_data_batch_size = images_labeled.shape[0]
        unlabeled_data_batch_size = images_unlabeled_weak_aug.shape[0]
        t_all_images = torch.cat([images_labeled, images_unlabeled_weak_aug, images_unlabeled_strong_aug])
        t_all_predictions = self.teacher(t_all_images)
        t_pred_labeled, t_pred_unlabeled_weak_aug, t_pred_unlabeled_strong_aug = torch.chunk(t_all_predictions,
                                                                                             [labeled_data_batch_size,
                                                                                              labeled_data_batch_size +
                                                                                              unlabeled_data_batch_size])
        t_loss_labeled = self.criterion(t_pred_labeled, targets)
        soft_pseudo_label = torch.softmax(t_pred_unlabeled_weak_aug.detach() / self.hparams.temperature, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(self.hparams.threshold).float()
        t_loss_unlabeled = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_pred_unlabeled_strong_aug, dim=-1)).sum(dim=-1) * mask
        )
        # FIXME: check if below line `batch_idx` is equivalent to
        #  `weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)`?
        weight_unlabeled = self.hparams.lambda_u * min(1.0, (batch_idx + 1) / self.hparams.uda_steps)
        t_loss_uda = t_loss_labeled + weight_unlabeled * t_loss_unlabeled

        s_all_images = torch.cat([images_labeled, images_unlabeled_strong_aug])
        s_all_predictions = self.student(s_all_images)
        s_pred_labeled, s_pred_unlabeled_strong_aug = torch.chunk(s_all_predictions, labeled_data_batch_size)

        s_loss_labeled_old = F.cross_entropy(s_pred_labeled.detach(), targets)
        s_loss = self.criterion(s_pred_unlabeled_strong_aug, hard_pseudo_label)
        opt_student.zero_grad()
        self.manual_backward(s_loss)
        opt_student.step()

        s_pred_labeled_new = self.student(images_labeled)
        s_loss_labeled_new = F.cross_entropy(s_pred_labeled_new.detach(), targets)
        # for `dot_product`, see explanation on https://github.com/google-research/google-research/issues/536
        dot_product = s_loss_labeled_old - s_loss_labeled_new
        _, hard_pseudo_label = torch.max(t_pred_unlabeled_strong_aug.detach(), dim=-1)
        t_loss_mpl = dot_product * F.cross_entropy(t_pred_unlabeled_strong_aug, hard_pseudo_label)
        t_loss = t_loss_uda + t_loss_mpl

        opt_teacher.zero_grad()
        self.manual_backward(t_loss)
        opt_teacher.step()
