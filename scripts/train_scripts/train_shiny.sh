#!/bin/bash

root_dir="data/shiny_blender/"
# list="helmet teapot toaster coffee ball car"
list="car"

for i in $list; do

python train.py --eval \
--render_mode defer+split_sum \
-s ${root_dir}${i} \
-m outputs/shiny/${i}/ \
-w --sh_degree -1 \
--lambda_predicted_normal 0.2 \
--lambda_zero_one 0.2 \
--env_res 512 \
--env_mode envmap \
--port 12991 \
--lambda_base_smoothness 0.02 \
--lambda_light_reg 0.001 \
--iterations 30000 \
--lambda_distortion 2000 \
--gaussian_type 2d \
--use_sdf \
--lambda_proj 5. \
--lambda_dev 1. \
--sphere_init

done
