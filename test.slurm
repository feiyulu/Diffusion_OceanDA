#!/bin/bash
#SBATCH -A cimes2
#SBATCH --qos=cimes-short
#SBATCH --job-name=train         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # memory per cpu-core
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=feiyu.lu@princeton.edu
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2021.11
conda activate pyqg

python main.py --work_path /scratch/cimes/feiyul/Diffusion_OceanDA/test2