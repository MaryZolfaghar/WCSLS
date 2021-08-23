#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn with batchsize=1, lr=1e-2 starts"

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 1 \
--N_cortical 2000 \
--bs_cortical 2 \
--lr_cortical 0.001 \
--order_ctx 'first' \
--truncated_mlp 'false' \
--out_file 'ctxF_results_rnn_hyperparams.P' \
--seed 0 \
--use_images \
--print_every 200 \
--N_episodic 1000 \
--bs_episodic 16 \
--lr_episodic 0.001 \
--cortical_task 'face_task' \
--checkpoints 50 \
--N_responses 'one' \
--N_contexts 2 \
--dimred_method 'pca' \