from voxel_grid import *
from point_cloud_FoV_utils import *
pcd_name = 'soldier'
participant = 'P01_V1'
trajectory_index = 0
pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
positions,orientations = get_point_cloud_user_trajectory(pcd_name=pcd_name,participant=participant)
position = positions[trajectory_index]
orientation = orientations[trajectory_index]
para_eye = [i*1024/1.8 for i in position]
para_eye[2] = -para_eye[2]
# para_eye = np.array(para_eye).reshape(3,1)
pitch_degree, yaw_degree, roll_degree = orientation
image_width, image_height = np.array([1920, 1080])
intrinsic_matrix = get_camera_intrinsic_matrix(image_width, image_height)
# Define camera extrinsic parameters (example values for rotation and translation)
extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, para_eye)



voxel_size = int(256/2)  # You can adjust this size as needed
min_bounds = np.array([-251,    0, -242])
max_bounds = np.array([ 262, 1023,  512])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, min_bounds, max_bounds)
line_sets_all_space = line_sets_from_voxel_grid_space(min_bounds, max_bounds, voxel_size)
# get the line sets's max bound
line_sets_max_bound = []
for line_set in line_sets_all_space:
    line_set_max_bound = line_set.get_max_bound()
    line_sets_max_bound.append(line_set_max_bound)
line_sets_max_bound = np.array(line_sets_max_bound)
max_bounds = np.max(line_sets_max_bound,axis=0)
# min_bound = np.min(line_sets_max_bound,axis=0)
print('max_bound:',max_bounds)
def line_sets_from_voxel_grid_space(min_bounds, max_bounds, voxel_size):
    line_sets = []
    for i in range(min_bounds[0], max_bounds[0], voxel_size):
        for j in range(min_bounds[1], max_bounds[1], voxel_size):
            for k in range(min_bounds[2], max_bounds[2], voxel_size):
                center = np.array([i, j, k]) + np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
                line_set = create_wireframe_cube(center, voxel_size)
                line_sets.append(line_set)
    return line_sets

 # evenly distrubit the points in the whole space and generate a new pcd
# pcd = randomly_add_points_in_point_cloud(N=1000,min_bound=min_bounds,max_bound=max_bounds)


# pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
# pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=8)


# add a coordinate frame at (0,0,0) min_bounds
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=min_bounds)
# add a sphere at the [0,500,500]
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8)
sphere.translate(para_eye)
# change color to red
sphere.paint_uniform_color([1,0,0])




o3d.visualization.draw_geometries([pcd,*line_sets_all_space,coordinate_frame])

# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, min_bounds, max_bounds)
# set the color of the voxel grid as the number of points in the voxel
# voxel_grid_color = np.array(voxel_grid.get_voxels()).shape[0]
# voxel_grid.paint_uniform_color([voxel_grid_color,0,0])
# Create a new point cloud to visualize the voxels
# colored_pcd = o3d.geometry.PointCloud()

# for voxel in voxel_grid.get_voxels():
#     # Assume you have a function to calculate the number of points in this voxel
#     num_points = np.random.randint(0, 100)  # Placeholder for actual count
#     color = np.array([num_points / 100.0, 0, 0])  # Normalize and map to red value
#     # Append the center of the voxel as a point in the new point cloud
#     colored_pcd.points.append(voxel.grid_index * voxel_grid.voxel_size)
#     colored_pcd.colors.append(color)
# o3d.visualization.draw_geometries([colored_pcd,*line_sets_all_space,coordinate_frame,sphere])

# o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,coordinate_frame,sphere])