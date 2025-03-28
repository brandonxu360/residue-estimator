#!/bin/bash
#SBATCH --partition=camas           # Partition to submit to
#SBATCH --job-name=unet_train       # Name of your job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --gres=gpu:2                # Request GPUs 
#SBATCH --mem=128G                  # Memory per node
#SBATCH --time=48:00:00             # Maximum run time (48 hours)
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Load modules
module load anaconda3 # or miniconda3

# Ensure required directories exist
mkdir -p logs

# Activate conda environment
ENV_NAME=${1:-ml-hpc-env}  # Use first argument or default to ml-hpc-env
source activate $ENV_NAME

# Capture start time
start_time=$(date +%s)

# Run training script
python -m src.scripts.train_multitask_unet \
    --data_root data/label \
    --job_id $SLURM_JOB_ID  # Pass job ID

# Calculate and log duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job $SLURM_JOB_ID completed in $duration seconds on nodes $SLURM_JOB_NODELIST."