#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp starts"

./scripts/local/run_mlp_corr.sh &
./scripts/local/run_mlp_ratio.sh &
./scripts/local/run_mlp_ttest.sh &
./scripts/local/run_mlp_regs.sh
wait

