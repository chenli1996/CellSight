import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from point_cloud_FoV_utils import *

# longdress_path = '../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1051.ply'
# pcd = o3d.io.read_point_cloud(longdress_path)

pcd = get_pcd_data(point_cloud_name='longdress', trajectory_index=0)
# labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)

# Define voxel size
voxel_size = int(256/2)  # You can adjust this size as needed
min_bounds = np.array([-251,    0, -242])
max_bounds = np.array([ 262, 1023,  512])
# (512+242)/128 = 5.891
# (262+251)/128 = 4.007
# (1023+0)/128 = 7.992
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, min_bounds, max_bounds)

# Get the points and their indices
points = np.asarray(pcd.points)
voxel_indices = np.floor(points / voxel_size).astype(int)
# Find unique indices and count the occurrences
unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)

# Combine indices and counts into a dictionary for easier access if needed
voxel_counts = {tuple(index): count for index, count in zip(unique_indices, counts)}

# get the voxel grid coordinates, which is the center of the voxel grid and voxel_grid_coords is a dict
voxel_grid_coords = {tuple(index): np.array(index) * voxel_size + voxel_size / 2 for index in voxel_counts.keys()}


# Function to create a wireframe cube at a given location with a given size
def create_wireframe_cube(center, size):
    # Vertices of the cube
    vertices = [
        center + np.array([-size / 2, -size / 2, -size / 2]),
        center + np.array([+size / 2, -size / 2, -size / 2]),
        center + np.array([-size / 2, +size / 2, -size / 2]),
        center + np.array([+size / 2, +size / 2, -size / 2]),
        center + np.array([-size / 2, -size / 2, +size / 2]),
        center + np.array([+size / 2, -size / 2, +size / 2]),
        center + np.array([-size / 2, +size / 2, +size / 2]),
        center + np.array([+size / 2, +size / 2, +size / 2])
    ]
    
    # Lines connecting the vertices
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Create line set with dashline
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines)
    )
    return line_set

# # Create line sets for each voxel
def line_sets_from_voxel_grid(voxel_grid):
    line_sets = []
    for voxel in voxel_grid.get_voxels():
        center = np.array(voxel_grid.get_voxel_center_coordinate(voxel.grid_index)) #* voxel_size + np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
        line_set = create_wireframe_cube(center, voxel_size)
        line_sets.append(line_set)
    return line_sets
line_sets = line_sets_from_voxel_grid(voxel_grid)

# # create line sets for the whole voxel grid space from (0,0,0) to (1024,1024,1024)
def line_sets_from_voxel_grid_space(min_bounds, max_bounds, voxel_size):
    line_sets = []
    for i in range(min_bounds[0], max_bounds[0], voxel_size):
        for j in range(min_bounds[1], max_bounds[1], voxel_size):
            for k in range(min_bounds[2], max_bounds[2], voxel_size):
                center = np.array([i, j, k]) + np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
                line_set = create_wireframe_cube(center, voxel_size)
                line_sets.append(line_set)
    return line_sets
line_sets_all_space = line_sets_from_voxel_grid_space(min_bounds, max_bounds, voxel_size)


# add a coordinate frame at (0,0,0) min_bounds
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=min_bounds)
# add a sphere at the [0,500,500]
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=16)
sphere.translate([0,500,500])
# change color to red
sphere.paint_uniform_color([1,0,0])



octree = voxel_grid.to_octree(max_depth=3)

# o3d.visualization.draw_geometries([voxel_grid, *line_sets, coordinate_frame])
# o3d.visualization.draw_geometries([voxel_grid, *line_sets_all_space, coordinate_frame])
# o3d.visualization.draw_geometries([octree,coordinate_frame, octree])
# o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,*line_sets,coordinate_frame])
o3d.visualization.draw_geometries([pcd,*line_sets_all_space,coordinate_frame,sphere])
# add two points in the pcd, one is the min_bound and the other is the max_bound
pcd.points.append(min_bounds)
pcd.points.append(max_bounds)
pcd.colors.append([1,0,0])
pcd.colors.append([0,1,0])
# o3d.visualization.draw_geometries([pcd,voxel_grid,*line_sets_all_space,coordinate_frame])
print('Done')
