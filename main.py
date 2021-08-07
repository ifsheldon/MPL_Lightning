import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch

import mpl_lightning


def configure_parser():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='experiment name')
    parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
    parser.add_argument('--gpu_num', default=1, type=int, help='training pu number')
    parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
    parser = mpl_lightning.datamodules.add_data_specific_args(parser)
    parser = mpl_lightning.LightningMPL.add_model_specific_args(parser)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def get_datamodule(args):
    if args.dataset == "cifar10":
        data_module = mpl_lightning.CIFAR10SSL_DM(
            args.data_path,
            args.batch_size,
            args.mu * args.batch_size,
            args.resize,
            args.num_labeled,
            args.expand_labels,
            args.eval_step,
            args.randaug,
            args.workers
        )
    else:
        data_module = mpl_lightning.CIFAR100SSL_DM(
            args.data_path,
            args.batch_size,
            args.mu * args.batch_size,
            args.resize,
            args.num_labeled,
            args.expand_labels,
            args.eval_step,
            args.randaug,
            args.workers
        )
    return data_module


def get_model(args):
    if args.dataset == "cifar10":
        depth, widen_factor, num_classes = 28, 2, 10
    else:
        depth, widen_factor, num_classes = 28, 8, 100
    mpl_model = mpl_lightning.LightningMPL(
        num_classes,
        depth,
        widen_factor,
        args.teacher_dropout,
        args.student_dropout,
        args.teacher_lr,
        args.student_lr,
        args.weight_decay,
        args.enable_nesterov,
        args.momentum,
        args.temperature,
        args.threshold,
        args.lambda_u,
        args.uda_steps,
        args.label_smoothing,
        args.warmup_steps,
        args.total_steps,
        args.student_wait_steps,
        args.ema,
        dropout=0.
    )
    return mpl_model


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    data_module = get_datamodule(args)
    model = get_model(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=2)
    print(f"hyperparameters = {args}")
    trainer = pl.Trainer(
        max_steps=args.total_steps,
        val_check_interval=args.eval_step,
        gpus=args.gpu_num,
        accelerator=None if args.gpu_num == 1 else "ddp",
        gradient_clip_val=args.grad_clip,
        auto_select_gpus=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data_module)
