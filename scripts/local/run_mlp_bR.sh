#!/usr/bin/env bash

# conda activate /Users/maryam/opt/anaconda3/env/csls

echo "Process mlp with
      hidden reps before the ReLU starts"

python main.py \
--cortical_model 'mlp' \
--before_ReLU \
--out_file 'results_mlp_bR.P' \
