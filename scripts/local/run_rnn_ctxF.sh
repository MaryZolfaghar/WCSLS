#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn starts"

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 20 \
--order_ctx 'first' \
--out_file 'ctxF_results_rnn.P' \
