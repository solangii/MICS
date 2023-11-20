#!/bin/bash

cd ../

nohup python -u run.py \
  -project 'mics' \
  -phase 'inc' \
  -dataset 'mini_imagenet' \
  -base_mode 'ft_cos' \
  -new_mode 'avg_cos' \
  -lr_new 0.5 \
  -epochs_new 5 \
  -gpu 0 \
  -temperature 16 \
  -model_dir 'your_model' \
  -dataroot 'your_dataroot' \
  -train mixup_hidden \
  -mixup_alpha 0.7 \
  -label_mix steep_dummy \
  -label_mix_threshold 0.1 \
  -normalized_middle_classifier False \
  -drop_last True \
  -st_ratio 0.3 \
  -use_resnet_alice True \
  > results/mini-inc.out &