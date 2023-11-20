#!/bin/bash

cd ../

nohup python -u run.py \
  -project 'mics' \
  -phase 'inc' \
  -dataset 'cub200' \
  -base_mode 'ft_cos' \
  -new_mode 'avg_cos' \
  -lr_new 0.1 \
  -epochs_new 20 \
  -gpu 2 \
  -temperature 16 \
  -model_dir "/home/solang/miil-conti/code/pretrained/cub200.pth" \
  -dataroot '/home/miil/Datasets/FSCIL-CEC' \
  -train mixup_hidden \
  -mixup_alpha 0.2 \
  -label_mix steep_dummy \
  -label_mix_threshold 0.3 \
  -normalized_middle_classifier False \
  -drop_last True \
  -st_ratio 0.3 \
  > results/cub200-inc.out &