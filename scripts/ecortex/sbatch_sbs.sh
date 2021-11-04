#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --output=run_rnn_ctxF_sbs_sbatch.out
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -c 1

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/csls

echo "Process sbatch rnn with batchsize=1, lr=15e-3 starts"

sbatch scripts/ecortex/run_rnn_ctxF_sbs.sh

wait
