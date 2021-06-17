#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn for all the analysis starts"

python main.py \
--cortical_model 'rnn' \
--out_file 'results_rnn' \
--analysis_type 'all' \
