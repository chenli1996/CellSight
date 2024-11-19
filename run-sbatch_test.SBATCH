#!/bin/bash
#SBATCH --account=pr_186_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=1GB	
#SBATCH --job-name=8i_p3_lr1e4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --output=./log/%j.out
data="${data}"
pred="${pred}"
echo "Data: ${data}"
echo "Pred: ${pred}"
exec > "./log/test${data}_${pred}_lr1e4_${SLURM_JOB_ID}.out" 2>&1
module purge

singularity exec --nv \
  --overlay /scratch/cl5089/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c \
		"source /ext3/env.sh; conda activate myenv; python run_sbatch_test.py --data "${data}" --pred "${pred}"" 