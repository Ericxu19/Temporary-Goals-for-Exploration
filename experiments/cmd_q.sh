#!/bin/bash -x
#SBATCH --ntasks=1 # Note that ntasks=1 runs multiple jobs in an array
#SBATCH --array=1-12%12
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -c 9
#SBATCH --mem=60G
#SBATCH -o %J_log.out 
#SBATCH --error=./job_%J.err
# run with sbatch path_to_cmd.sh path_to_cmd.txt path_to_results_folder

cmd_line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${1})
PYTHONPATH=./ $cmd_line --parent_folder $2 --num_envs 8
