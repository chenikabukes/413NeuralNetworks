# Run the following command to submit the job to the cluster
srun -p csc413 --gres gpu python3 -u train.py --wandb

# Check how many people using gpu
squeue -p csc413

srun -p csc413 --gres gpu python3 -u fastai_script.py --wandb --model 1

srun -p csc413 --gres gpu python3 -u train.py --model 1 --wandb

srun -p csc413 --gres gpu python3 -u train.py --model 1 --early_stopping