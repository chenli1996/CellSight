#!/bin/bash
#SBATCH --account=pr_186_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=128GB	
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=8i_p4_30_1e5_angular
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --output=./log/8i_p4_30_1e5_angular_%j.out
module purge
data="8i"
pred=4
singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c \
		"source /ext3/env.sh; conda activate myenv; python Cellsight_train_eval.py --data "${data}" --pred "${pred}"" 