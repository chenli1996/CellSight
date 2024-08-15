import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from point_cloud_FoV_utils import *
import pandas as pd

# longdress_path = '../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1051.ply'
# pcd = o3d.io.read_point_cloud(longdress_path)
def test():
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

def get_number_of_points_in_voxel_grid(pcd, voxel_size,min_bounds,max_bounds):
    # Get the points and their indices
    points = np.asarray(pcd.points).copy()
    points -= min_bounds
    voxel_indices = np.floor(points / voxel_size).astype(int)
    x_num,y_num,z_num = ((max_bounds - min_bounds + 1) / voxel_size).astype(int)
    # print('x_num:',x_num)
    # print('y_num:',y_num)
    # print('z_num:',z_num)
    # Find unique indices and count the occurrences
    # unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)


    # faster way on large dataset
    # # Convert the array to a DataFrame
    df = pd.DataFrame(voxel_indices)
    # Find unique rows and their counts
    # import pdb; pdb.set_trace()
    unique_df = df.groupby(df.columns.tolist()).size().reset_index(name='counts')
    # Extract unique_indices and counts
    unique_indices = unique_df[df.columns.tolist()].values
    counts = unique_df['counts'].values
    # assert unique_indices_new.all()==unique_indices.all()
    # assert counts_new.all()==counts.all()


    point_counts_in_voxel = {tuple(index): count for index, count in zip(unique_indices, counts)}
    # using a single integer index to represent the voxel index
    point_counts_in_voxel_integer = {index[0]*y_num*z_num + index[1]*z_num + index[2]: count for index, count in zip(unique_indices, counts)}
    # build a dict to map the integer index to the original index
    # integer_index_to_original_index = {index[0]*y_num*z_num + index[1]*z_num + index[2]: index for index in unique_indices}
    # print('integer_index_to_original_index:',integer_index_to_original_index)
    # get the voxel grid coordinates, which is the center of the voxel grid and voxel_grid_coords is a dict
    voxel_grid_coords = {tuple(index): np.array(index) * voxel_size + voxel_size / 2 + min_bounds for index in point_counts_in_voxel.keys()}
    # import pdb; pdb.set_trace()
    return point_counts_in_voxel_integer, voxel_grid_coords


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
def line_sets_from_voxel_grid(voxel_grid,voxel_size):
    line_sets = []
    for voxel in voxel_grid.get_voxels():
        # import pdb; pdb.set_trace()
        center = np.array(voxel_grid.get_voxel_center_coordinate(voxel.grid_index)) #* voxel_size + np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
        line_set = create_wireframe_cube(center, voxel_size)
        line_sets.append(line_set)
    return line_sets
# line_sets = line_sets_from_voxel_grid(voxel_grid)

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
# line_sets_all_space = line_sets_from_voxel_grid_space(min_bounds, max_bounds, voxel_size)


# # add a coordinate frame at (0,0,0) min_bounds
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=min_bounds)
# # add a sphere at the [0,500,500]
# sphere = o3d.geometry.TriangleMesh.create_sphere(radius=16)
# sphere.translate([0,500,500])
# # change color to red
# sphere.paint_uniform_color([1,0,0])



# octree = voxel_grid.to_octree(max_depth=3)

# o3d.visualization.draw_geometries([voxel_grid, *line_sets, coordinate_frame])
# o3d.visualization.draw_geometries([voxel_grid, *line_sets_all_space, coordinate_frame])
# o3d.visualization.draw_geometries([octree,coordinate_frame, octree])
# o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,*line_sets,coordinate_frame])
# o3d.visualization.draw_geometries([pcd,*line_sets_all_space,coordinate_frame,sphere])
# # add two points in the pcd, one is the min_bound and the other is the max_bound
# pcd.points.append(min_bounds)
# pcd.points.append(max_bounds)
# pcd.colors.append([1,0,0])
# pcd.colors.append([0,1,0])
# # o3d.visualization.draw_geometries([pcd,voxel_grid,*line_sets_all_space,coordinate_frame])
# print('Done')
def voxelizetion_para(voxel_size=256, min_bounds=np.array([-251,    0, -241]), max_bounds=np.array([ 262, 1023,  511])):
    # Define voxel size
    # voxel_size = int(256)  # You can adjust this size as needed
    # min_bounds = np.array([-251,    0, -241]) 
    # max_bounds = np.array([ 262, 1023,  511])
    # min_bounds = np.array([0,    0, 0]) 
    # max_bounds = np.array([ 1023, 1023,  1023]) 
    print('min_bounds:',min_bounds)
    print('max_bounds:',max_bounds)
    # print('voxel_grid voxel:',voxel_grid.get_voxels())
    # print('voxel_grid voxel:',len(voxel_grid.get_voxels()))
    # get the graph max bound after voxelization
    graph_voxel_grid_max_bound = (np.floor((max_bounds - min_bounds) / voxel_size)+1)*voxel_size + min_bounds-1
    graph_voxel_grid_min_bound = min_bounds
    # change to int
    graph_voxel_grid_max_bound = graph_voxel_grid_max_bound.astype(int)
    graph_voxel_grid_min_bound = graph_voxel_grid_min_bound.astype(int)
    print('graph max_bound:',graph_voxel_grid_max_bound)
    print('graph min_bound:',graph_voxel_grid_min_bound)

    pcd_N = randomly_add_points_in_point_cloud(
        N=100000,min_bound=graph_voxel_grid_min_bound,max_bound=graph_voxel_grid_max_bound)
    # pcd_N = evenly_add_points_in_point_cloud(
        # N=100,min_bound=graph_voxel_grid_min_bound,max_bound=graph_voxel_grid_max_bound)
    # get the voxel grid for the new pcd_N
    voxel_grid_N = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd_N, voxel_size, graph_voxel_grid_min_bound, graph_voxel_grid_max_bound)
    # print('voxel_grid voxel:',voxel_grid_N.get_voxels())
    # print('voxel_grid voxel index:',[voxel.grid_index for voxel in voxel_grid_N.get_voxels()])
    voxel_grid_index_set = set()
    for voxel in voxel_grid_N.get_voxels():
        voxel_grid_index_set.add(tuple(voxel.grid_index))
    # sort the set
    voxel_grid_index_set = sorted(voxel_grid_index_set) 
    # get the voxel grid coordinates, which is the center of the voxel grid and voxel_grid_coords is a dict
    voxel_grid_coords = {tuple(index): np.array(index) * voxel_size + voxel_size / 2 + graph_voxel_grid_min_bound for index in voxel_grid_index_set}
    voxel_grid_coords_array = np.array([voxel_grid_coords[index] for index in voxel_grid_index_set])


    # create a dict to map the integer index to the original index
    x_num,y_num,z_num = ((graph_voxel_grid_max_bound - graph_voxel_grid_min_bound + 1) / voxel_size).astype(int)
    original_index_to_integer_index = {index : index[0]*y_num*z_num + index[1]*z_num + index[2] for index in voxel_grid_index_set}
    print('original_index_to_integer_index:',original_index_to_integer_index)
    # convert the index to a singel integer index
    voxel_grid_integer_index_set = [index[0]*y_num*z_num + index[1]*z_num + index[2] for index in voxel_grid_index_set]
    # original index set
    voxel_grid_index_set = [index for index in voxel_grid_index_set]

    print('graph voxel index set:',voxel_grid_index_set)
    print('graph voxel integer index set:',voxel_grid_integer_index_set)
    print('length of graph voxel index:',len(voxel_grid_index_set))
    return {
        'graph_voxel_grid_max_bound': graph_voxel_grid_max_bound,
        'graph_voxel_grid_min_bound': graph_voxel_grid_min_bound,
        'graph_voxel_grid_integer_index_set': voxel_grid_integer_index_set,
        'graph_voxel_grid_index_set': voxel_grid_index_set,
        'graph_voxel_grid_coords': voxel_grid_coords,
        'graph_voxel_grid_coords_array': voxel_grid_coords_array,
        'original_index_to_integer_index': original_index_to_integer_index,
    }
#return graph_voxel_grid_max_bound,graph_voxel_grid_min_bound,voxel_grid_integer_index_set,voxel_grid_index_set,voxel_grid_coords,original_index_to_integer_index


def get_in_FoV_feature(graph_min_bound,graph_max_bound,voxel_size,intrinsic_matrix,extrinsic_matrix,image_width,image_height):
# get in-FoV feature
# uniformly distrubute the points in the whole space and generate a new pcd
    pcd_N = randomly_add_points_in_point_cloud(
        N=100000,min_bound=graph_min_bound,max_bound=graph_max_bound)
    point_counts_in_voxel_full, _ = get_number_of_points_in_voxel_grid(pcd_N,voxel_size,graph_min_bound,graph_max_bound)

    # get the points in the FoV
    pcd_N = get_points_in_FoV(pcd_N, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
    point_counts_in_voxel_FoV, _ = get_number_of_points_in_voxel_grid(pcd_N,voxel_size,graph_min_bound,graph_max_bound)

    in_FoV_voxel_percentage_dict = {}
    in_FoV_voxel_percentage_array = []
    for voxel_index in point_counts_in_voxel_full:
        if voxel_index in point_counts_in_voxel_FoV:
            in_FoV_voxel_percentage_dict[voxel_index] = point_counts_in_voxel_FoV[voxel_index]/point_counts_in_voxel_full[voxel_index]
            in_FoV_voxel_percentage_array.append(point_counts_in_voxel_FoV[voxel_index]/point_counts_in_voxel_full[voxel_index])
        else:
            in_FoV_voxel_percentage_dict[voxel_index] = 0
            in_FoV_voxel_percentage_array.append(0)
    # print('in_FoV_voxel_percentage_dict:',in_FoV_voxel_percentage_dict)
    # round the in_FoV voxel percentage to 2 decimal
    
    return in_FoV_voxel_percentage_dict,in_FoV_voxel_percentage_array,pcd_N

def get_occlusion_level_dict(pcd,para_eye,graph_min_bound,graph_max_bound,graph_voxel_grid_index_set,voxel_size,intrinsic_matrix,extrinsic_matrix,image_width,image_height):
    # pcd = pcd.voxel_down_sample(voxel_size=8)
    point_counts_in_voxel, _ = get_number_of_points_in_voxel_grid(pcd,voxel_size,graph_min_bound,graph_max_bound)
    # get the points in the FoV
    pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
    # pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=4)
    pcd_hpr = hidden_point_removal(pcd,para_eye)
    point_counts_in_voxel_hpr, _ = get_number_of_points_in_voxel_grid(pcd_hpr,voxel_size,graph_min_bound,graph_max_bound)
    occlusion_level_dict = {}
    occlusion_array = []
    for voxel_index in graph_voxel_grid_index_set:
        if voxel_index in point_counts_in_voxel_hpr:
            occlusion_level_dict[voxel_index] = point_counts_in_voxel_hpr[voxel_index]
            occlusion_array.append(point_counts_in_voxel_hpr[voxel_index]/point_counts_in_voxel[voxel_index])
            # occlusion_array.append(point_counts_in_voxel_hpr[voxel_index])# get the number of points in the voxel
            # if occlusion_array[-1] > 1: # this is only for ratio
                # print('occlusion_array:',occlusion_array[-1],voxel_index)
                # print('point_counts_in_voxel_hpr:',point_counts_in_voxel_hpr[voxel_index])
                # print('point_counts_in_voxel:',point_counts_in_voxel[voxel_index])
                # import pdb; pdb.set_trace()
        else:
            occlusion_level_dict[voxel_index] = 0
            occlusion_array.append(0)
    # print('occlusion_level_dict:',occlusion_level_dict)
    # round the occlusion level to 2 decimal
    occlusion_level_dict = {k: round(v,2) for k, v in occlusion_level_dict.items()}
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd_hpr])
    return occlusion_level_dict,occlusion_array,pcd_hpr

def get_occupancy_feature(pcd,graph_min_bound,graph_max_bound,graph_voxel_grid_index_set,voxel_size):
    point_counts_in_voxel, _ = get_number_of_points_in_voxel_grid(pcd,voxel_size,graph_min_bound,graph_max_bound)
    occupancy_dict = {}
    occupancy_array = []
    for voxel_index in graph_voxel_grid_index_set:
        if voxel_index in point_counts_in_voxel:
            occupancy_dict[voxel_index] = point_counts_in_voxel[voxel_index]
            occupancy_array.append(point_counts_in_voxel[voxel_index])
        else:
            occupancy_dict[voxel_index] = 0
            occupancy_array.append(0)
    return occupancy_dict,occupancy_array

# visualize the voxel grid
def visualize_voxel_grid(pcd,pcd_hpr,graph_min_bound,graph_max_bound,voxel_size,para_eye,voxel_grid_index_set,voxel_grid_coords):
    # add a coordinate frame at the min bound
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=graph_min_bound)
    # add a sphere at the eye position
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=8)
    sphere.translate(para_eye)
    # change color to red
    sphere.paint_uniform_color([1,0,0])
    # get the visualization of the voxel grid line set
    line_sets_all_space = line_sets_from_voxel_grid_space(graph_min_bound, graph_max_bound, voxel_size)
    
    #get the voxelized point cloud
    voxel_grid_hpr = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_hpr, voxel_size, graph_min_bound, graph_max_bound) 
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, graph_min_bound, graph_max_bound)
    line_sets_object = line_sets_from_voxel_grid(voxel_grid,voxel_size)

    # visualize the voxel grid
    # o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,coordinate_frame])
    o3d.visualization.draw_geometries([pcd,coordinate_frame,*line_sets_object])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd_hpr])
    # o3d.visualization.draw_geometries([voxel_grid])
    # o3d.visualization.draw_geometries([voxel_grid_hpr])
    


    # o3d.visualization.draw_geometries([voxel_grid,*line_sets_all_space,coordinate_frame,sphere])


    # fina all file which has string visualize_voxel_grid in the file content
