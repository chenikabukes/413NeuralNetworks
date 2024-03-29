#!/bin/bash
#SBATCH --job-name=my_job_name
#SBATCH --gres=gpu:t4:4  # Request 4 Tesla T4 GPU
#SBATCH --qos=m3         # Select the QOS. This will determine your max walltime
#SBATCH --time=04:00:00  # Job Duration (D-HH:MM:SS)
#SBATCH -c 10             # Number of CPU cores
#SBATCH --mem=10G        # Memory request
#SBATCH --output=cnn_transformer_output_%j.txt  # Standard output
#SBATCH --error=cnn_transformer_error_%j.txt    # Standard error

# Load any modules and activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Navigate to the script directory
cd ~/413NeuralNetworks

# Then run the script
python3 fastai_script.py \
 --wandb_name kai_ian_ci