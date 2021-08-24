#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn with batchsize=1, lr=15e-3 starts"
# N_cortical=4000, bs=1, lr_cortical 0.0015, train acc = 0.8125, test_acc = 0.8125
# N_cortical=2000, bs=1, lr_cortical 0.0015, train acc = 0.66875, test_acc = 0.84375
# N_cortical=4000, bs=1, lr_cortical 0.00025 , train acc = 0.6625, test_acc = 0.8125
# N_cortical=4000, bs=1, lr_cortical 0.002, train acc =  0.70625, test_acc = 0.875
# N_cortical=4000, bs=1, lr_cortical 0.00175, train acc = 0.7125 , test_acc = 0.6875
# N_cortical=4000, bs=1, lr_cortical 0.0005, train acc = 0.70625, test_acc = 0.84375
# N_cortical=4000, bs=1, lr_cortical 0.00075, train acc = 0.75625, test_acc = 0.84375
# N_cortical=10000, bs=1, lr_cortical 0015, train acc = 1, test_acc = 1
# N_cortical=7000, bs=1, lr_cortical 0015, train acc = 0.98, test_acc = 1
# N_cortical=6000, bs=1, lr_cortical 0015, train acc = 0.98, test_acc = 1
# N_cortical=5000, bs=1, lr_cortical 0015, train acc = 0.95625, test_acc = 1
# N_cortical=4500, bs=1, lr_cortical 0015, train acc = 0.8125, test_acc =  0.9375

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 10 \
--N_cortical 6000 \
--bs_cortical 1 \
--lr_cortical 0.0015 \
--order_ctx 'first' \
--sbs_analysis \
--truncated_mlp 'false' \
--out_file 'ctxF_results_rnn_sbs.P' \
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