# import open3d as o3d
# # import open3d as o3d
# print(o3d.__version__)

# vis = o3d.visualization.Visualizer()
# # vis.create_window() # the 0.17.0 version demands create_window() first, otherwise gives segmentation fault. Why?
# view_ctl = vis.get_view_control()
# print(view_ctl)
# cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
# assert id(ctr) == id(vis.get_view_control())  # assertion error.


import open3d
vis = open3d.visualization.Visualizer()
# vis.create_window() # the 0.17.0 version demands create_window() first, otherwise gives segmentation fault. Why?
ctr = vis.get_view_control() 
assert id(ctr) == id(vis.get_view_control())  # assertion error.