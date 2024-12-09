#!/bin/bash
#SBATCH -p QUEUE
#SBATCH --mem=15g
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user 
#SBATCH -J L+M-24-Shared-Task-Eval
#SBATCH -o slurm_outputs/scores/slurm-%A_%a.out

module load CUDA/11.3.0
module load cuDNN/8.2.1.32-CUDA-11.3.0

source /home/PATH_HERE/.bashrc
conda activate MolTextTranslationEval 

cd ~/MMLI_projects/Workshop/final_evaluation/molgen/

echo "Starting Main Script" 


python scorer.py
python scorer_combos.py

