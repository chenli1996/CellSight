#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB	
#SBATCH --job-name=fsvvd_angular_gen_tlr_mlp
#SBATCH --output=./log/generate_fsvvd_lr_tlr_mlp_angular_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --account=pr_186_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/cl5089/pytorch-example/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh ; conda activate env_py3_10_14 ; cd /scratch/$USER/point_cloud_FoV_Graph ; python node_feature_graph_FSVVD.py"		
