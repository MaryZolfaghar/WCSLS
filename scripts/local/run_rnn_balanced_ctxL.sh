#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 20 \
--order_ctx 'last' \
--truncated_mlp 'false' \
--balanced \
--seed 0 \
--use_images \
--print_every 200 \
--N_episodic 1000 \
--bs_episodic 16 \
--lr_episodic 0.001 \
--cortical_task 'wine_task' \
--N_cortical 1000 \
--bs_cortical 32 \
--lr_cortical 0.001 \
--checkpoints 50 \
--N_responses 'one' \
--N_contexts 2 \
--dimred_method 'pca' \
--out_file 'ctxL_results_rnn_balanced.P' \