#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp, wine task with two contexts starts"

python main.py \
--cortical_model 'mlp' \
--cortical_task 'wine_task' \
--N_contexts 4 \
--out_file 'results_mlp_wine_4ctx.P' \
