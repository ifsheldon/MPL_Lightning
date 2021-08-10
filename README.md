# MPL_Lightning
Lightning implementation of Meta Pseudo Label

The current working branch is `dev`.

## Features
This is intended to be a "minimum viable product" with following features:
* Important hyperparameters are specified as explicitly as possible, which means no confusing default values scattering around.
* As it's MVP, no profiling, no advanced metrics etc., you can add these if needed.
    * For more possible training features, please see [MPL-pytorch](https://github.com/kekmodel/MPL-pytorch)

## Run and train
To train a CIFAR10 model, run the following (Training time: ~2 days with one RTX-A6000)
```shell
python main.py --seed 5 --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --enable-nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8  --uda-steps 5000  --teacher-dropout 0.2 --student-dropout 0.2 --workers 16
```

## Performance

|                           |                          CIFAR10-4K                          |   SVHN-1K    | ImageNet-10% |
| :-----------------------: | :----------------------------------------------------------: | :----------: | :----------: |
|    Paper(w/ finetune)     |                         96.11 ± 0.07                         | 98.01 ± 0.07 |    73.89     |
| MPL-pytorch(w/o finetune) |                            94.46                             |      -       |      -       |
|  This code(w/o finetune)  | 92.83 ([learning curve](https://tensorboard.dev/experiment/EMgIOVOjQzSJkmwTApnu6w/#scalars)) |      -       |     TODO     |
| MPL-pytorch(w/ finetune)  |                      WIP in origin repo                      |      -       |      -       |
|  This code(w/ finetune)   |                             TODO                             |      -       |     TODO     |

Issues and some details about training in my code:

* It seems some of the training configurations given by MPL-pytorch do not match its published learning curve, see this [issue](https://github.com/kekmodel/MPL-pytorch/issues/20).
  * In the case of CIFAR10-4K without finetuning, I followed the setting of 300,000 steps of MPL-pytorch but without `--warmup-steps` and `--student-wait-steps` because after testing they don’t really matter. They will slow down approaching the same validation accuracy, instead, given the same number of training steps. However, `--uda-steps` matters after a bit testing.
* It seems some of the training configurations given by MPL-pytorch do not match those of the original paper training, see this [issue](https://github.com/kekmodel/MPL-pytorch/issues/15).
  * After inspecting the validation accuracy curve, it seems the model can do better if given more training steps, as the performance is being improved slowly but steadily in the last 100k of 300k steps.

## TODOs

* Fix `FIXME`s
* Finetuning

## Acknowledgements
* The codebase is based on [MPL-pytorch](https://github.com/kekmodel/MPL-pytorch) with some more comments, nicer variable names, so all the credits should go to MPL-pytorch and the original researchers at Google.
* Special thanks to tchaton@Github for his initial Lightning pseudo-code.