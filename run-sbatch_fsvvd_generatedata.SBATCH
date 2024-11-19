#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=64GB	
#SBATCH --job-name=8i_30lr_gen
#SBATCH --output=./log/generate_8i_lr_30_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --account=pr_186_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/cl5089/pytorch-example/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh ; conda activate env_py3_10_14 ; cd /scratch/$USER/point_cloud_FoV_Graph ; python node_feature_graph.py"		
