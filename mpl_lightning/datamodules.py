from typing import Optional

import pytorch_lightning as pl
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import math
from .datasets import CIFAR10SSL, CIFAR100SSL
from mpl_lightning.augmentation import RandAugment


def x_u_split(labels, num_labeled, num_classes, expand_labels, batch_size, eval_step):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    if expand_labels or num_labeled < batch_size:
        num_expand_x = math.ceil(batch_size * eval_step / num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, rand_aug, resize, mean, std):
        if rand_aug:
            n, m = rand_aug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=resize,
                                  padding=int(resize * 0.125),
                                  padding_mode='reflect')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=resize,
                                  padding=int(resize * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class CIFAR10SSL_DM(pl.LightningDataModule):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    def __init__(self,
                 data_path,
                 labeled_batch_size,
                 unlabeled_batch_size,
                 resize,
                 num_labeled,
                 num_classes,
                 expand_labels,
                 eval_step,
                 rand_aug,
                 workers
                 ):
        super(CIFAR10SSL_DM, self).__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        datasets.CIFAR10(self.hparams.data_path, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            base_dataset = datasets.CIFAR10(self.hparams.data_path, train=True, download=False)
            transform_labeled = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.hparams.resize,
                                      padding=int(self.hparams.resize * 0.125),
                                      padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cifar10_mean, std=self.cifar10_std)
            ])
            train_labeled_indices, train_unlabeled_indices = x_u_split(base_dataset.targets,
                                                                       self.hparams.num_labeled,
                                                                       self.hparams.num_classes,
                                                                       self.hparams.expand_labels,
                                                                       self.hparams.labeled_batch_size,
                                                                       self.hparams.eval_step)
            self.train_labeled_dataset = CIFAR10SSL(self.hparams.data_path,
                                                    train_labeled_indices,
                                                    train=True,
                                                    transform=transform_labeled)
            self.train_unlabeled_dataset = CIFAR10SSL(self.hparams.data_path,
                                                      train_unlabeled_indices,
                                                      train=True,
                                                      transform=TransformMPL(self.hparams.rand_aug,
                                                                             self.hparams.resize,
                                                                             CIFAR10SSL_DM.cifar10_mean,
                                                                             CIFAR10SSL_DM.cifar10_std))
        if stage in (None, "test"):
            transform_validation_set = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10SSL_DM.cifar10_mean, std=CIFAR10SSL_DM.cifar10_std)
            ])
            self.test_dataset = datasets.CIFAR10(self.hparams.data_path,
                                                 train=False,
                                                 transform=transform_validation_set,
                                                 download=False)

    def train_dataloader(self):
        # FIXME: shuffle?
        labeled_loader = torch.utils.data.DataLoader(self.train_labeled_dataset,
                                                     batch_size=self.hparams.labeled_batch_size,
                                                     num_workers=self.hparams.workers,
                                                     drop_last=True)
        unlabeled_loader = torch.utils.data.DataLoader(self.train_unlabeled_dataset,
                                                       batch_size=self.hparams.unlabeled_batch_size,
                                                       num_workers=self.hparams.workers,
                                                       drop_last=True)
        return {"labeled": labeled_loader, "unlabeled": unlabeled_loader}

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.hparams.labeled_batch_size,
                                           num_workers=self.hparams.workers)


class CIFAR100SSL_DM(pl.LightningDataModule):
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    def __init__(self,
                 data_path,
                 labeled_batch_size,
                 unlabeled_batch_size,
                 resize,
                 num_labeled,
                 num_classes,
                 expand_labels,
                 eval_step,
                 rand_aug,
                 workers
                 ):
        super(CIFAR100SSL_DM, self).__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        datasets.CIFAR100(self.hparams.data_path, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            base_dataset = datasets.CIFAR100(self.hparams.data_path, train=True, download=False)
            transform_labeled = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.hparams.resize,
                                      padding=int(self.hparams.resize * 0.125),
                                      padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
            ])
            train_labeled_indices, train_unlabeled_indices = x_u_split(base_dataset.targets,
                                                                       self.hparams.num_labeled,
                                                                       self.hparams.num_classes,
                                                                       self.hparams.expand_labels,
                                                                       self.hparams.labeled_batch_size,
                                                                       self.hparams.eval_step)
            self.train_labeled_dataset = CIFAR100SSL(self.hparams.data_path,
                                                     train_labeled_indices,
                                                     train=True,
                                                     transform=transform_labeled)
            self.train_unlabeled_dataset = CIFAR100SSL(self.hparams.data_path,
                                                       train_unlabeled_indices,
                                                       train=True,
                                                       transform=TransformMPL(self.hparams.rand_aug,
                                                                              self.hparams.resize,
                                                                              self.cifar100_mean,
                                                                              self.cifar100_std))
        if stage in (None, "test"):
            transform_validation_set = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
            ])
            self.test_dataset = datasets.CIFAR100(self.hparams.data_path,
                                                  train=False,
                                                  transform=transform_validation_set,
                                                  download=False)

    def train_dataloader(self):
        # FIXME: shuffle?
        labeled_loader = torch.utils.data.DataLoader(self.train_labeled_dataset,
                                                     batch_size=self.hparams.labeled_batch_size,
                                                     num_workers=self.hparams.workers,
                                                     drop_last=True)
        unlabeled_loader = torch.utils.data.DataLoader(self.train_unlabeled_dataset,
                                                       batch_size=self.hparams.unlabeled_batch_size,
                                                       num_workers=self.hparams.workers,
                                                       drop_last=True)
        return {"labeled": labeled_loader, "unlabeled": unlabeled_loader}

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.hparams.labeled_batch_size,
                                           num_workers=self.hparams.workers)
