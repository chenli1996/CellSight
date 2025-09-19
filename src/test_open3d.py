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
print(open3d.__version__)
vis = open3d.visualization.Visualizer()
vis.create_window() # the 0.17.0 version demands create_window() first, otherwise gives segmentation fault. Why?
ctr = vis.get_view_control() 
print(ctr)
vis.poll_events()
vis.update_renderer()
vis.capture_depth_float_buffer(do_render=True)
# vis.capture_screen_back_buffer('open3d_test_results/test.png')
assert id(ctr) == id(vis.get_view_control())  # assertion error.