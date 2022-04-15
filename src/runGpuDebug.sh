#!/bin/bash
#SBATCH --job-name=ANVAEdebug
#SBATCH --account=PAA0023
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpudebug
  
 
cd $SLURM_SUBMIT_DIR
 
module load python/3.6-conda5.2 cuda/11.0.3

source activate local
conda info --envs
which python
python run_tf_vae.py
