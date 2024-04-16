import open3d
vis = open3d.visualization.Visualizer()
# vis.create_window() # the 0.17.0 version demands create_window() first, otherwise gives segmentation fault. Why?
ctr = vis.get_view_control() 
assert id(ctr) == id(vis.get_view_control())  # assertion error.