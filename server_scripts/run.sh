#!/bin/bash

#SBATCH --job-name=urb

#SBATCH --partition=rknodes
#SBATCH --qos=big_bonk

#SBATCH --time=23:59:00

## SBATCH --nodes=1
## SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1



PATH_PROGRAM="/home/$USER/repos/URB" # Path to URB directory
PUT_PROGRAM_TO="/app" # Where to mount in a container

PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo2.sif" # Container image location

# Script to be run inside container and outputs location (note: inside binded URB -> also binded already)
CMD_PATH="$PATH_PROGRAM/server_scripts/cmd_container.sh" 
PRINTS_SAVE_PATH="$PATH_PROGRAM/server_scripts/container_job_printouts/output_$SLURM_JOB_ID.all"

mkdir -p "$(dirname "$PRINTS_SAVE_PATH")"
# Run container by adding code by binding, run commands from cmd_container.sh, save printouts to a file
singularity exec --nv --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" > "$PRINTS_SAVE_PATH" 2>&1