#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp with two responses starts"

python main.py \
--cortical_model 'mlp' \
--N_responses 'two' \
--out_file 'results_mlp_2resps.P' \
