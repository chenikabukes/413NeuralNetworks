# Run the following command to submit the job to the cluster
srun -p csc401 --gres gpu python3 -u train.py

# Check how many people using gpu
squeue -p csc401