#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
#SBATCH --output=R-mlp-con-bR.%j.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/csls

echo "Process mlp with continuous regression and
      hidden reps before the ReLU starts"

python main.py \
--before_ReLU \
--out_file 'results_mlp_con_bR.P' \
