# close all python processes except this one
# kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}')
kill -9 $(ps | grep '[p]ython' | awk '{print $1}')

# move folder ./data to ../point_cloud_FoV_Graph
mv ./data ../point_cloud_FoV_Graph

# copy *9060* to ssh greene.hpc.nyu.edu:/scratch/cl5089/point_cloud_FoV_Graph/data/soldier_VS128_LR
scp ./*9060* greene.hpc.nyu.edu:/scratch/cl5089/point_cloud_FoV_Graph/data/soldier_VS128_LR

# run nvidia-smi and watch every 0.1s
watch -n 0.1 nvidia-smi

# srun
srun  --nodes=1  --ntasks-per-node=1 --cpus-per-task=4 --time=2:00:00 --mem=64G --gres=gpu:h100:1 --pty /bin/bash