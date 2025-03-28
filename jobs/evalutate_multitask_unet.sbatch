#!/bin/bash
#SBATCH --partition=camas           # Partition to submit to
#SBATCH --job-name=unet_test       # Name of your job
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

# MODEL_JOB_ID: required
# ENV_NAME: optional (default: ml-hpc-env)
if [ -z "$1" ]; then
  echo "Error: MODEL_JOB_ID argument is required."
  echo "Usage: sbatch your_script.sh <MODEL_JOB_ID> [ENV_NAME]"
  exit 1
fi
MODEL_JOB_ID=$1
ENV_NAME=${2:-ml-hpc-env}

# Activate conda environment
source activate $ENV_NAME

# Capture start time
start_time=$(date +%s)

# Run testing script
python -m src.scripts.evaluate_multitask_unet \
    --data_root data/label \
    --model_path results/unet/best_model_${MODEL_JOB_ID}.pth

# Calculate and log duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job $SLURM_JOB_ID completed in $duration seconds on nodes $SLURM_JOB_NODELIST."