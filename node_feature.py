from voxel_grid import *
from point_cloud_FoV_utils import *


pcd_name = 'soldier'
participant = 'P01_V1'
trajectory_index = 0

# Load the point cloud data
pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
# get the position and orientation for the given participant and trajectory index
positions,orientations = get_point_cloud_user_trajectory(pcd_name=pcd_name,participant=participant)
position = positions[trajectory_index]
orientation = orientations[trajectory_index]
para_eye = [i*1024/1.8 for i in position]
para_eye[2] = -para_eye[2]
# para_eye = np.array(para_eye).reshape(3,1)
pitch_degree, yaw_degree, roll_degree = orientation
image_width, image_height = np.array([1920, 1080])
# Define camera intrinsic parameters
intrinsic_matrix = get_camera_intrinsic_matrix(image_width, image_height)
# Define camera extrinsic parameters
extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, para_eye)

# generate graph voxel grid features
voxel_size = int(256*2)
min_bounds = np.array([-251,    0, -241]) 
max_bounds = np.array([ 262, 1023,  511])
# get the graph max and min bounds
graph_max_bound,graph_min_bound,graph_voxel_grid_index_set = voxelizetion_para(
    voxel_size=voxel_size, min_bounds=min_bounds, max_bounds=max_bounds)
# uniformly distrubute the points in the whole space and generate a new pcd
pcd_N = randomly_add_points_in_point_cloud(
    N=80000,min_bound=graph_min_bound,max_bound=graph_max_bound)
# get the points in the FoV
pcd_N = get_points_in_FoV(pcd_N, intrinsic_matrix, extrinsic_matrix, image_width, image_height)

# get the points in the FoV
# pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
# remove the hidden points
# pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=1)
# get the # of ponts in voxel grid
# point_counts_in_voxel, voxel_grid_coords = get_number_of_points_in_voxel_grid(pcd_N,voxel_size,graph_min_bound)
point_counts_in_voxel, voxel_grid_coords = get_number_of_points_in_voxel_grid(pcd,voxel_size,graph_min_bound)
print('point_counts_in_voxel:',point_counts_in_voxel)
print('length of point_counts_in_voxel:',len(point_counts_in_voxel))
print('voxel_grid_coords:',voxel_grid_coords)

for voxel_index in graph_voxel_grid_index_set:
    if voxel_index in point_counts_in_voxel:
        print('voxel_index:',voxel_index)
        print('point_counts_in_voxel:',point_counts_in_voxel[voxel_index])
    else:
        print('voxel_index:',voxel_index)
        print('point_counts_in_voxel:',0)


# add a coordinate frame at the min bound
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=graph_min_bound)
# add a sphere at the eye position
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8)
sphere.translate(para_eye)
# change color to red
sphere.paint_uniform_color([1,0,0])
# get the visualization of the voxel grid line set
line_sets_all_space = line_sets_from_voxel_grid_space(graph_min_bound, graph_max_bound, voxel_size)
# visualize the voxel grid
o3d.visualization.draw_geometries([pcd,*line_sets_all_space,coordinate_frame])
# o3d.visualization.draw_geometries([colored_pcd,*line_sets_all_space,coordinate_frame,sphere])
# o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,coordinate_frame,sphere])