#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp starts"

./scripts/local/run_rnn_corr.sh &
./scripts/local/run_rnn_ratio.sh &
./scripts/local/run_rnn_ttest.sh &
./scripts/local/run_rnn_regs.sh
wait

