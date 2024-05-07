# close all python processes except this one
# kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}')
kill -9 $(ps | grep '[p]ython' | awk '{print $1}')

# move folder ./data to ../point_cloud_FoV_Graph
mv ./data ../point_cloud_FoV_Graph

# copy *9060* to ssh greene.hpc.nyu.edu:/scratch/cl5089/point_cloud_FoV_Graph/data/soldier_VS128_LR
scp ./*9060* greene.hpc.nyu.edu:/scratch/cl5089/point_cloud_FoV_Graph/data/soldier_VS128_LR