#!/bin/bash
#SBATCH -p QUEUE
#SBATCH --mem=15g
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user 
#SBATCH -J LPM-24-Dataset-Process
#SBATCH -o slurm_outputs/scores/slurm-%A_%a.out

module load CUDA/11.3.0
module load cuDNN/8.2.1.32-CUDA-11.3.0

source /home/a-m/USERNAME/.bashrc
conda activate MolTextTranslationEval 

cd ~/PATH_HERE/final_evaluation/captioning/

echo "Starting Main Script" 


python extract_props.py 
python scorer_props.py

