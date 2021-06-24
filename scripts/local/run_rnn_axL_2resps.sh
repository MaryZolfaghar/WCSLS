#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn with two responses starts"

python main.py \
--cortical_model 'rnn' \
--N_cortical 2000 \
--order_ax 'last' \
--N_responses 'two' \
--out_file 'axL_results_rnn_2resps.P' \
