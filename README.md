# MPL_Lightning
Lightning implementation of Meta Pseudo Label

The current working branch is `dev`.

## Features
This is intended to be a "minimum viable product" with following features:
* Important hyperparameters are specified as explicitly as possible, which means no confusing default values scattering around.
* As it's MVP, no profiling, no advanced metrics etc., you can add these if needed.
    * For more possible training features, please see [MPL-pytorch](https://github.com/kekmodel/MPL-pytorch)

## Run and train
To train a CIFAR10 model, run the following
```shell
python main.py --seed 5 --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --enable-nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --workers 16
```

## TODOs
* Fix `FIXME`s
* Finetuning

## Acknowledgements
* The codebase is based on [MPL-pytorch](https://github.com/kekmodel/MPL-pytorch) with some more comments, nicer variable names, so all the credits should go to MPL-pytorch and the original researchers at Google.
* Special thanks to tchaton@Github for his initial Lightning pseudo-code.