#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp starts"

python main.py \
--cortical_model 'stepwisemlp' \
--nruns_cortical 20 \
--out_file 'results_stepwisemlp.P' \
