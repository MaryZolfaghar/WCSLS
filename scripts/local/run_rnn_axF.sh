#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn starts"

python main.py \
--cortical_model 'rnn' \
--order_ax 'first' \
--out_file 'axF_results_rnn.P' \
