#!/bin/bash

root_dir="data/tensoir_synthetic/"
# list="armadillo lego ficus hotdog"

list="lego"

for i in $list; do

python train.py --eval \
--render_mode defer+split_sum \
-s ${root_dir}${i} \
-m outputs/tir/${i}/ \
-w --sh_degree -1 \
--lambda_predicted_normal 0.2 \
--lambda_zero_one 0.4 \
--env_res 512 \
--env_mode envmap \
--port 12991 \
--lambda_brdf_smoothness 0.02 \
--lambda_distortion 2000 \
--gaussian_type 2d \
--use_sdf \
--densify_grad_threshold 0.0002 \
--lambda_proj 5. \
--lambda_dev 1. \
--sphere_init

done
