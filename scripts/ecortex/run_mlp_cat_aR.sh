#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
#SBATCH --output=R-mlp-cat-aR.%j.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/csls

echo "Process mlp with categorical regression and
      hidden reps after the ReLU starts"

python main.py \
--cortical_model 'mlp'
--out_file 'results_mlp_cat_aR.P' \
