#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process stepwise mlp starts"

python main.py \
--cortical_model 'stepwisemlp' \
--nruns_cortical 20 \
--out_file 'results_stepwisemlp.P' \
# defaults
--seed 0 \
--print_every 200 \
--N_episodic 1000 \
--bs_episodic 16 \
--lr_episodic 0.001 \
--cortical_task 'face_task' \
--N_cortical 1000\
--bs_cortical 32 \ 
--lr_cortical 0.001 \ 
--checkpoints 50 \
--order_ctx 'first' \
--N_responses 'one' \
--N_contexts 2 \
