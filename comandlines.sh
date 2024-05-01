# close all python processes except this one
# kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}')
kill -9 $(ps | grep '[p]ython' | awk '{print $1}')

# move folder ./data to ../point_cloud_FoV_Graph
mv ./data ../point_cloud_FoV_Graph