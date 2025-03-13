#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=4GB	
#SBATCH --job-name=rendering_fsvvd
#SBATCH --output=./log/rendering_fsvvd%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cl5089@nyu.edu
#SBATCH --account=pr_186_tandon_advanced

module purge

singularity exec --nv \
    --overlay /scratch/cl5089/pytorch-example/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh ; conda activate env_py3_10_14 ; cd /scratch/$USER/point_cloud_FoV_Graph ; Xvfb :100 -screen 0 1920x1080x24 & export DISPLAY=:100; python align_FSVVD_rendering.py"		
