#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --output=sbatch_main_analysis.out
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -c 1

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/csls

echo "Process sbatch main analysis"

sbatch scripts/ecortex/run_rnn_ctxF_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.1_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.2_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.3_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.4_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.5_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.6_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.7_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.8_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion0.9_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion1_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion10_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion30_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion50_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion70_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion90_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion500_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion1000_ecortex.sh &
sbatch scripts/ecortex/run_rnn_ctxF_lesion10000_ecortex.sh &

sbatch scripts/ecortex/run_mlp_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.1_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.2_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.3_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.4_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.5_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.6_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.7_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.8_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion0.9_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion1_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion10_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion30_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion50_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion70_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion90_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion500_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion1000_ecortex.sh &
sbatch scripts/ecortex/run_mlp_lesion10000_ecortex.sh &

wait
