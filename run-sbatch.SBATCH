#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=256GB	
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=graphgru
#SBATCH --output=./log/slurm_p1_90_30_G2_full_lre4_dim100_vs64_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate myenv; python graphgru.py" 