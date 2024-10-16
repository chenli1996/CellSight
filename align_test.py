import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
pcd = o3d.io.read_point_cloud('../point_cloud_data/processed_FSVVD/FSVVD_300/Chatting/Raw/0_binary.ply')
vis.add_geometry(pcd)
view_ctl = vis.get_view_control()
print(vis)
print(view_ctl)
