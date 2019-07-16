#!/bin/bash
#SBATCH --account=def-dkulic
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --output=./job_scripts_output/dqn_n_step_CartPole-v0_0_9_%N-%j.out        # %N for node name, %j for jobID
## Main processing command
module load cuda cudnn 
source ~/tf_gpu/bin/activate
python ./train_cartpole.py  --env_name CartPole-v0 --seed 0 --n_step 9 --exp_name dqn_n_step_CartPole-v0_9step_s0