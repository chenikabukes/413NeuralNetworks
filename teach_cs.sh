# Run the following command to submit the job to the cluster
srun -p csc401 --gres gpu python3 -u train.py --wandb

# Check how many people using gpu
squeue -p csc401

srun -p csc401 --gres gpu python3 -u fastai_script.py --wandb
