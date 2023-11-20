#!/bin/bash

cd ../

nohup python -u run.py \
  -project 'mics' \
  -phase 'inc' \
  -dataset 'cifar100' \
  -base_mode 'ft_cos' \
  -new_mode 'avg_cos' \
  -lr_new 0.0005 \
  -epochs_new 10 \
  -gpu 0 \
  -temperature 16 \
  -model_dir 'your_model' \
  -dataroot 'your_dataroot' \
  -train mixup_hidden \
  -mixup_alpha 0.5 \
  -label_mix steep_dummy \
  -label_mix_threshold 0.5 \
  -normalized_middle_classifier False \
  -drop_last True \
  -st_ratio 0.01 \
  -use_resnet_alice False \
  > results/cifar100-inc.out &
