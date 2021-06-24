#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn with two responses starts"

python main.py \
--cortical_model 'rnn' \
--N_cortical 2000 \
--order_ax 'first' \
--N_responses 'two' \
--out_file 'axF_results_rnn_2resps.P' \
