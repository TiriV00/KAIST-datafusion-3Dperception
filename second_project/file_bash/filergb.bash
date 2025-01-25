#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 24:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 10


#< Charge resources to account 
#SBATCH --account T_2022_DLFEA

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

#choose the correct path on the machine
python ./rgbmain.py

conda deactivate