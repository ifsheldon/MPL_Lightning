import math
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as metrics
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from .aux_modules import SmoothCrossEntropy, ModelEMA
from .aux_modules import WideResNet


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.shape[0]
    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, max_k).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class LightningMPL(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("LightningMPL")
        parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
        parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
        parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning rate')
        parser.add_argument('--student_lr', default=0.01, type=float, help='train learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
        parser.add_argument('--enable-nesterov', action='store_true', help='use nesterov')
        parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
        parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
        parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
        parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
        parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
        parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
        parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
        parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
        parser.add_argument('--ema', default=0.0, type=float, help="EMA decay rate")
        return parent_parser

    def __init__(self,
                 num_classes,
                 depth,
                 widen_factor,
                 teacher_dropout,
                 student_dropout,
                 teacher_lr,
                 student_lr,
                 weight_decay,
                 enable_nesterov,
                 momentum,
                 temperature,
                 threshold,
                 lambda_u,
                 uda_steps,
                 label_smoothing,
                 warmup_steps,
                 total_steps,
                 student_wait_steps,
                 ema,
                 dropout=0.,
                 ):
        """
        Init MPL
        :param num_classes: number of classes
        :param depth: model depth
        :param widen_factor: widen factor of WideResNet
        :param teacher_dropout: dropout on last dense layer of teacher
        :param student_dropout: dropout on last dense layer of student
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
        :param warmup_steps: warmup steps
        :param total_steps: number of total steps to run
        :param student_wait_steps: student warmup steps
        :param ema: EMA decay rate of weight average of student
        :param dropout: per-layer dropout
        """
        super(LightningMPL, self).__init__()
        self.save_hyperparameters()
        self.teacher = WideResNet(num_classes=num_classes,
                                  depth=depth,
                                  widen_factor=widen_factor,
                                  dropout=dropout,
                                  dense_dropout=teacher_dropout)
        self.student = WideResNet(num_classes=num_classes,
                                  depth=depth,
                                  widen_factor=widen_factor,
                                  dropout=dropout,
                                  dense_dropout=student_dropout)
        self.enable_student_ema = ema > 0.0
        if self.enable_student_ema:
            self.avg_student_model = ModelEMA(self.student, ema)
        else:
            self.avg_student_model = None
        self.criterion = self.create_loss_fn()
        # activate manual optimization
        self.automatic_optimization = False
        # metrics
        self.train_metrics = nn.ModuleDict({
            "train/top1_acc": metrics.Accuracy(top_k=1),
            "train/top5_acc": metrics.Accuracy(top_k=5),
        })
        self.validation_metrics = nn.ModuleDict({
            "val/top1_acc": metrics.Accuracy(top_k=1, compute_on_step=False),
            "val/top5_acc": metrics.Accuracy(top_k=5, compute_on_step=False),
        })
        self.step = 0  # fixme: distributed training problem?

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
        t_scheduler = get_cosine_schedule_with_warmup(t_optimizer, self.hparams.warmup_steps,
                                                      self.hparams.total_steps)
        s_optimizer = optim.SGD(student_parameters,
                                lr=self.hparams.student_lr,
                                momentum=self.hparams.momentum,
                                nesterov=self.hparams.enable_nesterov)
        s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                      self.hparams.warmup_steps,
                                                      self.hparams.total_steps,
                                                      num_wait_steps=self.hparams.student_wait_steps)
        return ({"optimizer": t_optimizer, "lr_scheduler": t_scheduler},
                {"optimizer": s_optimizer, "lr_scheduler": s_scheduler})

    def forward(self, image_batch):
        return self.avg_student_model(image_batch) if self.enable_student_ema else self.student(image_batch)

    def training_step(self, batch, batch_idx):
        opt_teacher, opt_student = self.optimizers()
        sch_teacher, sch_student = self.lr_schedulers()
        images_labeled, targets = batch["labeled"]
        (images_unlabeled_weak_aug, images_unlabeled_strong_aug), _target = batch["unlabeled"]
        targets = targets.long()
        labeled_data_batch_size = images_labeled.shape[0]
        unlabeled_data_batch_size = images_unlabeled_weak_aug.shape[0]
        t_all_images = torch.cat([images_labeled, images_unlabeled_weak_aug, images_unlabeled_strong_aug])
        t_all_predictions = self.teacher(t_all_images)
        t_pred_labeled, t_pred_unlabeled_weak_aug, t_pred_unlabeled_strong_aug = torch.split(t_all_predictions,
                                                                                             [labeled_data_batch_size,
                                                                                              unlabeled_data_batch_size,
                                                                                              unlabeled_data_batch_size],
                                                                                             dim=0)
        t_loss_labeled = self.criterion(t_pred_labeled, targets)
        soft_pseudo_label = torch.softmax(t_pred_unlabeled_weak_aug.detach() / self.hparams.temperature, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(self.hparams.threshold).float()
        t_loss_unlabeled = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_pred_unlabeled_strong_aug, dim=-1)).sum(dim=-1) * mask
        )
        weight_unlabeled = self.hparams.lambda_u * min(1.0, (self.step + 1) / self.hparams.uda_steps)
        t_loss_uda = t_loss_labeled + weight_unlabeled * t_loss_unlabeled

        s_all_images = torch.cat([images_labeled, images_unlabeled_strong_aug])
        s_all_predictions = self.student(s_all_images)
        s_pred_labeled, s_pred_unlabeled_strong_aug = torch.split(s_all_predictions,
                                                                  [labeled_data_batch_size, unlabeled_data_batch_size])

        s_loss_labeled_old = F.cross_entropy(s_pred_labeled.detach(), targets)
        s_loss = self.criterion(s_pred_unlabeled_strong_aug, hard_pseudo_label)

        opt_student.zero_grad()
        self.manual_backward(s_loss)
        opt_student.step()
        sch_student.step()

        with torch.no_grad():
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
        sch_teacher.step()

        return {
            "teacher_loss": t_loss.detach(),
            "student_loss": s_loss.detach(),
            "s_pred_labeled": s_pred_labeled_new.detach(),
            "targets": targets
        }

    def training_step_end(self, step_outputs):
        if self.enable_student_ema:
            self.avg_student_model.update_parameters(self.student)
        self.log("train/teacher_loss", step_outputs["teacher_loss"], prog_bar=True, logger=True)
        self.log("train/student_loss", step_outputs["student_loss"], prog_bar=True, logger=True)
        self.log("step", self.step, prog_bar=True, logger=False)
        s_pred_labeled = step_outputs["s_pred_labeled"]
        targets = step_outputs["targets"]
        for metric_name in self.train_metrics:
            metric = self.train_metrics[metric_name]
            metric(s_pred_labeled, targets)
            self.log(metric_name, metric)

        self.step += 1

    def validation_step(self, batch, batch_idx):
        eval_model = self.avg_student_model if self.enable_student_ema else self.student
        imgs, targets = batch
        pred = eval_model(imgs)
        loss = self.criterion(pred, targets)
        return {
            "loss": loss,
            "preds": pred,
            "targets": targets
        }

    def validation_step_end(self, step_output):
        loss = step_output["loss"]
        self.log("val/loss", loss, prog_bar=True)
        preds = step_output["preds"]
        targets = step_output["targets"]
        for metric_name in self.validation_metrics:
            metric = self.validation_metrics[metric_name]
            metric(preds, targets)

    def validation_epoch_end(self, _outputs):
        for metric_name in self.validation_metrics:
            metric = self.validation_metrics[metric_name]
            self.log(metric_name, metric.compute())
