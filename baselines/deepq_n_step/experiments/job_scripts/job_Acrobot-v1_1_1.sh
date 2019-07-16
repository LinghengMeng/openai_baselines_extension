#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dqn_n_step_Acrobot-v1_1_1_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn 
source ~/tf_gpu/bin/activate
python ./train_cartpole.py  --env_name Acrobot-v1 --seed 1 --n_step 1 --exp_name dqn_n_step_Acrobot-v1_1step_s1