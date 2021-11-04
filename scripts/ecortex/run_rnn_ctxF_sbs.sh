#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --output=run_rnn_ctxF_sbs_e1.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/csls

echo "Process rnn with batchsize=1, lr=15e-3 starts"
# --------------------------------------------------------------------------------
# N_cortical=4000, bs=1, lr_cortical 0.0015, train acc = 0.8125, test_acc = 0.8125
# N_cortical=2000, bs=1, lr_cortical 0.0015, train acc = 0.66875, test_acc = 0.84375
# N_cortical=4000, bs=1, lr_cortical 0.00025 , train acc = 0.6625, test_acc = 0.8125
# N_cortical=4000, bs=1, lr_cortical 0.002, train acc =  0.70625, test_acc = 0.875
# N_cortical=4000, bs=1, lr_cortical 0.00175, train acc = 0.7125 , test_acc = 0.6875
# N_cortical=4000, bs=1, lr_cortical 0.0005, train acc = 0.70625, test_acc = 0.84375
# N_cortical=4000, bs=1, lr_cortical 0.00075, train acc = 0.75625, test_acc = 0.84375
# N_cortical=10000, bs=1, lr_cortical 0015, train acc = 1, test_acc = 1
# N_cortical=7000, bs=1, lr_cortical 0015, train acc = 0.98, test_acc = 1
# N_cortical=5000, bs=1, lr_cortical 0015, train acc = 0.95625, test_acc = 1
# N_cortical=4500, bs=1, lr_cortical 0015, train acc = 0.8125, test_acc =  0.9375
# --------------------------------------------------------------------------------
# *Final*: N_cortical=8000, bs=1, lr_cortical 0015, train acc = 0.98, test_acc = 1
# sbs_every = 1, print_every = 300
# ~4 min every 300 steps if doing the analysis at every 1 step  (sbs_every=1)
# for N_cortical = 8000 it'll be ~27-28 300-steps (print_every=300) per run
# total ~ time for each run = 27/28*4min = 108/112 min
# total ~ time for 20 runs = 112*20 = 2240 ~ 37 hrs 
# ~ 2 hrs each run, 20 run = 40 hrs
# --------------------------------------------------------------------------------

python main.py \
--cortical_model 'rnn' \
--nruns_cortical 20 \
--sbs_every 1 \
--N_cortical 8000 \
--bs_cortical 1 \
--lr_cortical 0.0015 \
--order_ctx 'first' \
--sbs_analysis \
--truncated_mlp 'false' \
--out_file 'ctxF_results_rnn_sbs_e1.P' \
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
