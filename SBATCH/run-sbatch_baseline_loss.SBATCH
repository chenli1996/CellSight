#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=128GB	
#SBATCH --job-name=fsvvd_8i_baseline
#SBATCH --output=./log/fsvvd_8i_baseline_loss_lr30_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --account=pr_186_tandon_advanced

module purge

singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c \
		"source /ext3/env.sh; conda activate myenv; python baseline_loss.py" 
