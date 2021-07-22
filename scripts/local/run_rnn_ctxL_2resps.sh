#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn with two responses starts"

python main.py \
--cortical_model 'rnn' \
--N_cortical 2000 \
--nruns_cortical 20 \
--order_ctx 'last' \
--N_responses 'two' \
--out_file 'ctxL_results_rnn_2resps.P' \
