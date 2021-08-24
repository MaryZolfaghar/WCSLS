#!/usr/bin/env bash

# conda activate /Users/Maryam/anaconda3/envs/csls
# /Users/Maryam/anaconda3/envs/csls

echo "Process rnn starts - ctx first - lesion p 10000 - measuring norm of grad."

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 20 \
--order_ctx 'first' \
--is_lesion \
--lesion_p 10000 \
--measure_grad_norm \
--out_file 'ctxF_results_rnn_lesionp10000_normgrad.P' \
--seed 0 \
--truncated_mlp 'false' \
--use_images \
--print_every 200 \
--N_episodic 1000 \
--bs_episodic 16 \
--lr_episodic 0.001 \
--cortical_task 'face_task' \
--N_cortical 1000 \
--bs_cortical 32 \
--lr_cortical 0.001 \
--checkpoints 50 \
--N_responses 'one' \
--N_contexts 2 \
--dimred_method 'pca' \