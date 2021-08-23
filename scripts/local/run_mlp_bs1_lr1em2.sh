#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process mlp with batchsize=2"
# N_cortical=2000, lr_cortical=1e-02, train acc : 0.5, test acc: 0.5
# N_cortical=2000, lr_cortical=1e-03, train acc : 0.83125, test acc: 0.75
# N_cortical=2000, lr_cortical=1e-04, train acc : 0.65, test acc: 0.65625
# N_cortical=2000, lr_cortical=1e-05, train acc : 0.5 , test acc: 0.5
# N_cortical=2000, lr_cortical=1e-06, train acc : 0.5, test acc: 0.5
# N_cortical=2000, lr_cortical=1e-07, train acc : 0.5, test acc: 0.5


python main.py \
--cortical_model 'mlp' \
--nruns_cortical 2 \
--N_cortical 2000 \
--bs_cortical 2 \
--lr_cortical 0.01 \
--truncated_mlp 'false' \
--out_file 'results_mlp_hyperparams.P' \
--seed 0 \
--use_images \
--print_every 200 \
--N_episodic 1000 \
--bs_episodic 16 \
--lr_episodic 0.001 \
--cortical_task 'face_task' \
--checkpoints 50 \
--order_ctx 'first' \
--N_responses 'one' \
--N_contexts 2 \
--dimred_method 'pca' \
