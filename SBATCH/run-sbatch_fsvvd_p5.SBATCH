#!/bin/bash
#SBATCH --account=pr_186_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --mem=128GB	
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=1_1e6_fsvvd_p5_angular
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --output=./log/fsvvd_p5_1_1e6_angular_%j.out
module purge
data="fsvvd_raw"
pred=5
singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c \
		"source /ext3/env.sh; conda activate myenv; python Cellsight_train_eval.py --data "${data}" --pred "${pred}"" 