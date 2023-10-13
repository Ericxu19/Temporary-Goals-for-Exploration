#!/bin/bash -x
#SBATCH --ntasks=1 # Note that ntasks=1 runs multiple jobs in an array
#SBATCH --array=1-12%12
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o ./slurm_output/%J.out # this is where the output goes

# run with sbatch path_to_cmd.sh path_to_cmd.txt path_to_results_folder

cmd_line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${1})
PYTHONPATH=./ $cmd_line --parent_folder $2 --num_envs 4 --checkpoint_dir /checkpoint/${USER}/$((SLURM_ARRAY_TASK_ID + SLURM_ARRAY_JOB_ID))
