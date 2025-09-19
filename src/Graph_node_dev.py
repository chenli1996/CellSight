import open3d as o3d
import numpy as np
# Load or create point cloud
trajectory_index = 0
point_cloud_name = 'longdress'
if point_cloud_name == 'longdress':
    point_cloud_path = '8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,147]))#longdress
# pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")

# Define octree
octree = o3d.geometry.Octree(max_depth=4)  # Choose an appropriate depth based on your needs
octree.convert_from_point_cloud(pcd, size_expand=0.01)  # size_expand can help avoid edge cases

# Define the bounding box for the query (0<x<16, 0<y<16, 0<z<16)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(16, 16, 16))
print('bbox',bbox)
# Query points within the bounding box
points_in_box = octree.locate_leaf_node(bbox).point_indices

# Get number of points
# num_points = len(points_in_box)
# print("Number of points in the specified tile:", num_points)
