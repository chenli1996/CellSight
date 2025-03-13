#!/bin/bash
#SBATCH --account=pr_186_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=256GB	
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=1_f_p4_angular
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --output=./log/fsvvd_p4_1_10_angular_%j.out
module purge
data="fsvvd_raw"
pred=4
singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c \
		"source /ext3/env.sh; conda activate myenv; python Cellsight_train_eval.py --data "${data}" --pred "${pred}"" 