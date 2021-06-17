#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp for ttest analysis starts"

python main.py \
--cortical_model 'mlp' \
--out_file 'results_mlp' \
--analysis_type 'ttest' \
