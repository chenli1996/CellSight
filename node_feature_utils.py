import open3d as o3d
import warnings
import pdb
import os
import numpy as np
import open3d as o3d
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd



# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])
import numpy as np
import os
from collections import defaultdict

## get all cell occupied 
def get_all_cell_occupied(base_path = '/Users/chenli/research/point cloud/tile_data/recons/',frame_index_off_set = 1051,frame_index_begin = 0,frame_index_end = 149):
    # Assuming 'base_path' is the directory containing the dataset
    # base_path = '/Users/chenli/research/point cloud/tile_data/recons/'
    # frame_index_off_set = 1051
    # frame_index_begin = 0  # The frame index you're interested in
    # frame_index_end = 149#1350
    occupied_cell = defaultdict(set)
    for frame_index in range(frame_index_begin+frame_index_off_set,frame_index_end+frame_index_off_set+1):
        # print(frame_index)
        # for x in range(16):
        #     for y in range(16):
        #         for z in range(16):
        #             file_name = f'{frame_index:04d}/longdress_{frame_index:04d}_{x:02d}_{y:02d}_{z:02d}_{1}.ply'
        #             file_path = os.path.join(base_path, file_name)
        #             if os.path.isfile(file_path):                    
        #                 occupied_cell[frame_index].add(tuple([x,y,z]))
        file_frame_index = (frame_index - frame_index_off_set)%150 + frame_index_off_set
        DIRECTORY = base_path + str(file_frame_index)
        for filename in sorted(os.listdir(DIRECTORY)):
            filename_s = filename.split('_')
            if len(filename_s)<4:
                continue
            x,y,z = int(filename_s[2]),int(filename_s[3]),int(filename_s[4])
            occupied_cell[frame_index-frame_index_off_set].add(tuple([x,y,z]))
    return occupied_cell
# len(occupied_cell)
# occupied_cell


# # get trajectory of user

import csv

# def parse_trajectory_data(file_path='./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv',user_index='P01_V1'):
#     positions = []
#     orientations = []
#     with open(file_path, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#         next(csvreader)  # Skip the header
#         for row in csvreader:
#             # Assuming the CSV columns are ordered as mentioned
#             # import pdb;pdb.set_trace()
#             if row[7] == user_index:                
#                 positions.append([float(row[1]), float(row[2]), float(row[3])])
#                 orientations.append([float(row[4]), float(row[5]), float(row[6])])
#     return np.array(positions), np.array(orientations)

def parse_trajectory_data(file_path='./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv', user_index='P01_V1'):
    df = pd.read_csv(file_path)
    user_data = df[df.iloc[:, 7] == user_index]
    positions = user_data.iloc[:, 1:4].values
    orientations = user_data.iloc[:, 4:7].values
    return positions, orientations
# Function to convert Euler angles to direction vector (simplified)
def euler_to_direction(yaw, pitch, roll):
    # Assuming yaw (Z), pitch (Y), roll (X) in radians for simplicity
    # direction = np.array([
    #     -np.sin(yaw) * np.cos(pitch),
    #     -np.sin(pitch),
    #     np.cos(yaw) * np.cos(pitch),
        
    # ])
    direction = np.array([
        -np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
        -np.cos(yaw) * np.cos(pitch),
        
    ])    
    return direction


# usage:
# trajectory_positions, trajectory_orientations = parse_trajectory_data("./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv",user_index='P01_V1')

# Function to check if a point is inside the frustum
def is_point_in_frustum(point, camera_pos, camera_dir, fov_x, aspect_ratio, near_clip, far_clip):
    # Calculate the frustum planes
    half_fov_x = np.radians(fov_x / 2)
    half_fov_y = np.arctan(np.tan(half_fov_x) / aspect_ratio)
    
    # Direction vectors to the right and up based on camera orientation
    right_vec = np.cross(camera_dir, np.array([0, 1, 0]))
    up_vec = np.cross(right_vec, camera_dir)
    
    # Calculate near and far plane centers
    near_center = camera_pos + camera_dir * near_clip
    far_center = camera_pos + camera_dir * far_clip
    
    # Check if point is between near and far planes
    if np.dot(camera_dir, point - near_center) < 0 or np.dot(camera_dir, point - far_center) > 0:
        return False
    
    # Check if point is inside the horizontal field of view
    to_point_dir = point - camera_pos
    to_point_proj_right = np.dot(to_point_dir, right_vec)
    to_point_proj_forward = np.dot(to_point_dir, camera_dir)
    if abs(np.arctan(to_point_proj_right / to_point_proj_forward)) > half_fov_x:
        return False
    
    # Check if point is inside the vertical field of view
    to_point_proj_up = np.dot(to_point_dir, up_vec)
    if abs(np.arctan(to_point_proj_up / to_point_proj_forward)) > half_fov_y:
        return False
    
    return True

## get all cell in the view frustum
def get_cell_in_view_frustum(trajectory_positions, trajectory_orientations, occupied_cell):
    # trajectory_positions, trajectory_orientations = parse_trajectory_data("./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv",user_index='P01_V1')
    fov_x = 60  # Field of view in X dimension
    aspect_ratio = 1024 / 768  # Aspect ratio
    near_clip = 0.1  # Near clipping plane
    far_clip = 2000.0  # Far clipping plane



    grid_size = 64
    count = 0
    in_count = 0
    FoV_cell = defaultdict(set)
    for trajectory_index in range(trajectory_positions.shape[0]):  
        count += 1
        selected_position = trajectory_positions[trajectory_index]
        para_eye = [i*1000/1.75 for i in selected_position] # 1.75mm for each point in 8i dataset


        # get yaw pitch roll and to orientation
        selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
        pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary
        # Calculate the direction vector from the orientation
        orientation = euler_to_direction(yaw, pitch, roll)

        # para_lookat = [para_eye[i]+orientation[i] for i in range(3)]
        # print('para_lookat',para_lookat)
        if count %1000 == 0:
            print(count)
            # print('camera eye',para_eye)
            # print('orientation',orientation)
        # print(selected_position,selected_orientation)

        for cell in occupied_cell[trajectory_index%150]:
            cell_center = np.array([(cell[0]+0.5)*grid_size,(cell[1]+0.5)*grid_size,(cell[2]+0.5)*grid_size])  # Assuming cell represents the center
            # print(cell_center)
            if is_point_in_frustum(cell_center, para_eye, orientation, fov_x, aspect_ratio, near_clip, far_clip):
                FoV_cell[trajectory_index].add(cell)
                # print(f"Cell {cell} is inside the frustum.")
                in_count += 1
                pass
            else:
                # print(f"Cell {cell} is outside the frustum.")
                pass
    return FoV_cell
            
            
def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    """
    Check if a ray intersects with a box (AABB).
    :param ray_origin: Origin of the ray (np.array)
    :param ray_direction: Direction of the ray (np.array)
    :param box_min: Minimum vertex of the box (np.array)
    :param box_max: Maximum vertex of the box (np.array)
    :return: Boolean indicating if there is an intersection
    """
    t_min = (box_min - ray_origin) / ray_direction
    t_max = (box_max - ray_origin) / ray_direction
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_near = np.max(t1)
    t_far = np.min(t2)

    if t_near > t_far or t_far < 0:
        return False

    return True

def occlusion_count_S_and_R(viewpoint, target_cell_center, FoV_cell_index, target_cell_index, cell_size,dense_n_of_point):
    """
    Count the number of surrounding cells that meet the occlusion criteria.
    :param viewpoint: Viewpoint coordinates and orientation (X, Y, Z, yaw, pitch, roll)
    :param target_cell_center: Center of the target cell 'c'
    :param cell_size: Size of each cell
    :param target_distance: Distance from the viewpoint to the target cell 'c'
    :return: Number of cells occluding the target cell 'c'
    """
    R_c_max = 0
    target_cell_point_number = dense_n_of_point[target_cell_index]
    count = 0
    ray_origin = np.array(viewpoint)
    target_cell_center = np.array(target_cell_center)
    ray_direction = target_cell_center - ray_origin  # Assumes already normalized if needed
    ray_direction /= np.linalg.norm(ray_direction)   # Normalize the ray direction
    target_distance = np.linalg.norm(ray_origin-target_cell_center)
    # Define the bounds of the surrounding cells
#     offsets = [-1,0,1]
    my_list = []
    range_len = 5
    offsets = range(-range_len,range_len+1)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                # if target_cell_index == tuple([7, 8, 8]) and dx==-1 and dy==1 and dz==-3:
                    # import pdb;pdb.set_trace()
                # if target_cell_index == tuple([7, 8, 8]):
                    # if ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
                        # import pdb;pdb.set_trace()                              
                        # my_list.append([dx,dy,dz])
                        # if  neighbor_cell == tuple([6,9,5]):
                        #     print('ww')                     
                
                neighbor_cell = tuple([dx+target_cell_index[0],dy+target_cell_index[1],dz+target_cell_index[2]])
                if dx == dy == dz == 0 or neighbor_cell not in FoV_cell_index:
                    continue  # Skip the center cell itself

                neighbor_center = target_cell_center + np.array([dx, dy, dz])*cell_size
                box_min = neighbor_center - np.array([cell_size/2])
                box_max = neighbor_center + np.array([cell_size/2])
                distance_to_neighbor = np.linalg.norm(neighbor_center - ray_origin)
                # import pdb;pdb.set_trace()
              
                if distance_to_neighbor < target_distance and ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
                    count += 1
                    # if target_cell_index == tuple([6, 11, 8]) and neighbor_cell == tuple([6,11,6]):
                    #     import pdb;pdb.set_trace()
                    
                    
                    R_c = dense_n_of_point[neighbor_cell]
                    if R_c > R_c_max:
                        R_c_max = R_c
#     if target_cell_index == tuple([7, 11, 8]):
    # if target_cell_index == tuple([7, 8, 8]):
        # print(my_list,"*****")
        
#         following code execute all pair-wise checking for whether we have occlusion or not


#     for neighbor_cell in FoV_cell_index:
#         if neighbor_cell == target_cell_index:
#             continue
#         # neighbor_center = target_cell_center + np.array([dx, dy, dz])
#         neighbor_center = np.array(neighbor_cell)*cell_size+ np.array([cell_size/2])
#         box_min = neighbor_center - np.array([cell_size/2])
#         box_max = neighbor_center + np.array([cell_size/2])
#         distance_to_neighbor = np.linalg.norm(neighbor_center - ray_origin)
#         # if target_cell_index == tuple([7,8,8]):
#             # import pdb;pdb.set_trace()
#         # if target_cell_index == tuple([6, 11, 8]) and neighbor_cell == tuple([6,11,7]):
#             # import pdb;pdb.set_trace()             
#         if distance_to_neighbor < target_distance and ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
#             count += 1
#             if target_cell_index == tuple([7,8,8]):
#                 import pdb;pdb.set_trace()
#                 # w
#             R_c = dense_n_of_point[neighbor_cell]
#             if R_c > R_c_max:
#                 R_c_max = R_c         
        
        
    return count,R_c_max/target_cell_point_number


def occlusion_mapping(R_c, S_c):
    """
    Maps R(c) to an integer value based on given thresholds.

    :param R_c:   The R(c) value to map
    :param alpha: List or tuple of alpha values [α0, α1, α2]
    :param beta:  The beta value
    :param S_c:   The S(c) value
    :return:      The mapped integer value
    """
    # R_c += 0.1
    alpha0 = 0.6
    alpha1=1
    alpha2=3
    beta = 0.8
    
    alpha = [alpha0,alpha1,alpha2]
    
    thresholds = [a * (beta ** (S_c - 1)) for a in alpha]
    # print(thresholds)

    if R_c < thresholds[0]:
        return 0
    elif thresholds[0] <= R_c < thresholds[1]:
        return 1
    elif thresholds[1] <= R_c < thresholds[2]:
        return 2
    elif thresholds[2] <= R_c:
        return 3

def occlusion_mapping_continous(R, S, a=0.1, b=1):
    return np.exp(-a * S - b * R)
    
    # Example usage:
# alpha_values = [alpha0, alpha1, alpha2]  # Replace with actual values for α0, α1, and α2
# beta_value = beta  # Replace with the actual value for β
# S_c_value = S_c  # Replace with the actual value for S(c)
# R_c_value = R_c  # Replace with the actual value for R(c)

# result = occlusion_mapping(R_c_value, alpha_values, beta_value, S_c_value)
# print(result)
            
## get feature for all cell    

import pickle
import pandas as pd


def get_occlusion_feature_for_all_cell(FoV_cell,occupied_cell,trajectory_positions,trajectory_orientations,trajectory_frame_index_begin,trajectory_frame_index_end,grid_size=64):
    filepath_n_of_point = '/Users/chenli/research/point cloud/tile_data/number.pkl'
    with open(filepath_n_of_point, 'rb') as f:
        N_of_point = pickle.load(f)


    # alpha_values = [alpha0, alpha1, alpha2]  # Replace with actual values for α0, α1, and α2
    # beta_value = beta  # Replace with the actual value for β

    # # frame_index = 1051
    # frame_index_begin = 1051  # The frame index you're interested in
    # frame_index_end = 1051#1350
    full_feature_data = []
    count = 0
    for trajectory_index in range(trajectory_frame_index_begin,trajectory_frame_index_end+1):  
        selected_position = trajectory_positions[trajectory_index]
        para_eye = [i*1000/1.75 for i in selected_position] # 1.75mm for each point in 8i dataset
        # print('camera eye',para_eye)
        # get yaw pitch roll and to orientation
        selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
        pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary
        # Calculate the direction vector from the orientation
        orientation = euler_to_direction(yaw, pitch, roll)
        # print('orientation',orientation)
        # print(selected_position,selected_orientation)
        for cell in occupied_cell[trajectory_index]:
            count += 1
            
            if cell in FoV_cell[trajectory_index]:
                cell_center = np.array([(cell[0]+0.5)*grid_size,(cell[1]+0.5)*grid_size,(cell[2]+0.5)*grid_size])  # Assuming cell represents the center
                dense_n_of_point = N_of_point[trajectory_index%150,:,:,:,0]
                S_c_value,R_c_value = occlusion_count_S_and_R(np.array(para_eye), cell_center, FoV_cell[trajectory_index], cell, grid_size,dense_n_of_point)  # 
                occlusion_result = occlusion_mapping_continous(R_c_value, S_c_value)
                # print(occlusion_result,S_c_value,R_c_value,cell_center,cell)

                full_feature_data.append([cell,trajectory_index,occlusion_result])
            else:
                full_feature_data.append([cell,trajectory_index,0])
            # u = cell2node[cell]
            # offsets = [-1, 0, 1]
            # for dx in offsets:
            #     for dy in offsets:
            #         for dz in offsets:
            #             neighbor_cell = tuple([dx+cell[0],dy+cell[1],dz+cell[2]])
            #             if dx == dy == dz == 0 or neighbor_cell not in FoV_cell[trajectory_index]:
            #                 continue                    
            #             i = cell2node[neighbor_cell]
            # # [u,i,trajectory_index,result,count]
            #             full_feature_data.append([u, i, trajectory_index, result, count])#!!!!!!!!!!!!!!!!!!!!
            if count %10000==0:
                print(count)
            #     # break
            #     # w
            #     print(count)
    df = pd.DataFrame(full_feature_data, columns=['cell', 'ts', 'label'])
    return df


def cell_occlusion_level_distribution(node_occlusion_feature_df,label='label'):
    print(node_occlusion_feature_df[label].describe())
    # Plotting histogram of 'label'
#     node_occlusion_feature_df['label'].plot.hist(bins=10, alpha=0.6)  # Adjust bins as needed
#     plt.title('Distribution of label')
#     plt.xlabel('Label')
#     plt.ylabel('Frequency')
#     plt.show()
    
#     node_occlusion_feature_df.boxplot(column=['label'])
#     plt.title('Boxplot of label')
#     plt.ylabel('Label')
#     plt.show()
    
#     node_occlusion_feature_df['label'].plot.density()
#     plt.title('Density Plot of label')
#     plt.xlabel('Label')
#     plt.show()
    
#     stats.probplot(node_occlusion_feature_df['label'], dist="norm", plot=plt)
#     plt.title('Q-Q Plot of label against Normal distribution')
#     plt.show()


    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

    # Histogram
    axs[0, 0].hist(node_occlusion_feature_df[label], bins=10, alpha=0.6)
    axs[0, 0].set_title('Distribution of '+label)
    axs[0, 0].set_xlabel(label)
    axs[0, 0].set_ylabel('Frequency')

    # Boxplot
    node_occlusion_feature_df.boxplot(column=[label], ax=axs[0, 1])
    axs[0, 1].set_title('Boxplot of '+label)
    axs[0, 1].set_ylabel(label)

    # Density Plot
    node_occlusion_feature_df[label].plot(kind='density', ax=axs[1, 0])
    axs[1, 0].set_title('Density Plot of '+label)
    axs[1, 0].set_xlabel('Label')

    # Q-Q Plot
    stats.probplot(node_occlusion_feature_df[label], dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title('Q-Q Plot of label against Normal distribution')

    plt.tight_layout()  # Adjust layout to make room for titles
    plt.show()

from collections import defaultdict
def cell2node2cell():
    cell2node = defaultdict(int)
    node2cell = defaultdict(tuple)
    for y in range(16):
        for z in range(16):
            for x in range(16):
                cell2node[tuple([x,y,z])] = y*256 + z* 16 + x
                node2cell[y*256 + z* 16 + x] = tuple([x,y,z])
    return cell2node,node2cell
    
def generate_event_for_tgn(node_occlusion_feature_df,occupied_cell,cell2node):
    event_data = []
    count = 0
    for index,row in node_occlusion_feature_df.iterrows():
        # print(index,row)
        cell = row['cell']
        ts = row['ts']
        label = row['label']
        u = cell2node[cell]
        offsets = [-1, 0, 1]
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    neighbor_cell = tuple([dx+cell[0],dy+cell[1],dz+cell[2]])
                    if dx == dy == dz == 0 or neighbor_cell not in occupied_cell[ts]:
                        continue                    
                    i = cell2node[neighbor_cell]
                    count += 1
                    event_data.append([u, i, ts, label, count])
                    if count %100000==0:
                        print(count)
            # import pdb;pdb.set_trace()
            # break 
    
    event_data_df = pd.DataFrame(event_data, columns=['u', 'i', 'ts', 'label','idx'])        
    return event_data_df




import os
import numpy as np
import open3d as o3d


def add_heatmap_on_pcd(pcd,heatmap_value):
    original_colors = np.asarray(pcd.colors)
    # original_colors.shape
    heatmap_colors = np.full(original_colors.shape, heatmap_value)
    # # Blend original colors with heatmap colors
    # # This blending formula can be adjusted according to your needs
    blended_colors =  heatmap_colors
    blended_colors = (original_colors/10 + heatmap_colors) / 2
    pcd.colors = o3d.utility.Vector3dVector(blended_colors)
    return pcd


def rendering_from_node_feature(trajectory_positions,trajectory_orientations,df,FoV_cell,trajectory_index,frame_index_off_set):    
    # get XYZ data
    selected_position = trajectory_positions[trajectory_index]
    para_eye = [i*1000/1.75 for i in selected_position]
    # print(para_eye)

    pcd_list = []

    # get yaw pitch roll and to orientation
    selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
    pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary
    # Calculate the direction vector from the orientation
    orientation = euler_to_direction(yaw, pitch, roll)
    # print('orientation',orientation)

    # para_eye = [200,500,1000]
    # para_eye = [200,1024,1000]
    # orientation = [0,0,-1]
    para_lookat = [para_eye[i]+orientation[i] for i in range(3)]
    # print(para_lookat)
    cell_count = 0
    # label_u = set(df[df['label'] <= occlusion_threshold]['cell'])
    # label1_u = set(df[df['label'] == 1]['cell'])
    # label2_u = set(df[df['label'] == 2]['cell'])
    # label3_u = set(df[df['label'] == 3]['cell'])
    # print('0',label_u)
    # print('1',label1_u)
    # print('2',label2_u)
    # print('3',label3_u)    
    DIRECTORY = './tile_data/recons/'+str(trajectory_index%150+frame_index_off_set)
    # print(label3_u)
    
    for filename in sorted(os.listdir(DIRECTORY)):
        filename_s = filename.split('_')
        # print(filename)
        if len(filename_s)<4:
            continue
        if filename[-5] != '6':
            continue
        cell = tuple([int(filename_s[2]),int(filename_s[3]),int(filename_s[4])])    
        # if cell not in FoV_cell[trajectory_index]:
        #     heatmap_value = 0
        # else:
        # heatmap_value = df[df.cell==cell].iloc[0].label
            
        # import pdb;pdb.set_trace()
        # if str(cell) in label3_u or str(cell) in label2_u or str(cell) in label1_u:
        # if cell in label_u:# or cell in label2_u or cell in label1_u:            
        #     # print('*')
        #     continue


            
        tile_ply = os.path.join(DIRECTORY, filename)
        pcd = o3d.io.read_point_cloud(tile_ply) 
        
        # if heatmap_value<0.5:
            # import pdb;pdb.set_trace()
        # pcd = add_heatmap_on_pcd(pcd,heatmap_value)
        pcd_list.append(pcd)
        cell_count += 1
    # print(cell_count)
    
    
#     *********

# the following code is adding a cube to render the location of that cube
# Define the cube dimensions
#     cube_size = 64
#     num_points = 100000  # Increase the number of points for a denser point cloud

#     # Generate points within a cube centered at the origin
#     points = np.random.uniform(-cube_size / 2, cube_size / 2, size=(num_points, 3))

#     grid_size = 64
#     # Translate points to shift the center of the cube to (100, 100, 100)
#     my_x,my_y,my_z = 7,8,8# index of tile x,y,z
#     center = np.array([(my_x+0.5)*grid_size, (my_y+0.5)*grid_size, (my_z+0.5)*grid_size])
#     points = points + center

#     # Create a PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Optionally, color the points based on their Z coordinate
#     colors = np.zeros_like(points)
#     colors[:, 2] = (points[:, 2] - center[2] + cube_size / 2) / cube_size  # Blue gradient based on Z coordinate
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     pcd_list.append(pcd)

# #**********
#     hidden point removel
    _, pt_map = pcd_list.hidden_point_removal(para_eye, 100)
    # print("Visualize result")
    pcd_new = pcd_list.select_by_index(pt_map)

# ------------------------
    
    
    # pcd_list = [o3d.io.read_point_cloud("./8i/longdress/longdress/Ply/longdress_vox10_1051.ply")]
    o3d.visualization.draw(pcd_new,lookat=para_lookat,eye=para_eye,up=[0., 1, 0.])
    
import numpy as np

def linear_regression(x, y):
    """
    Computes the coefficients of a linear regression y = mx + c using least squares.
    
    Args:
    - x: numpy array of shape (n,), the independent variable
    - y: numpy array of shape (n,), the dependent variable
    
    Returns:
    - m: Slope of the fitted line
    - c: Intercept of the fitted line
    """
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def predict_next_state_tlp(user_data, window_size=2,dof=6,future_steps = 1):
    """
    Predicts the next state based on the last 'window_size' states using linear regression.
    
    Args:
    - user_data: numpy array of shape (n, 6), where n is the number of timesteps,
                 and 6 represents the 6 DoF (x, y, z, yaw, pitch, roll).
    - window_size: int, the number of states to consider for the prediction
    
    Returns:
    - next_state: numpy array of shape (6,), representing the predicted next state.
    """
    if user_data.shape[0] < window_size:
        raise ValueError("Not enough data for prediction.")
    
    next_state = np.zeros(dof)
    time_steps = np.arange(window_size)
    
    # Perform linear regression on each DoF using the last 'window_size' states
    for i in range(dof):
        m, c = linear_regression(time_steps, user_data[-window_size:, i])
        next_state[i] = m * (window_size+future_steps-1) + c  # Predict the next state
    
    return next_state

def predict_next_state_tlp_rad(user_data, window_size=2,dof=6,future_steps = 1):
    """
    Predicts the next state based on the last 'window_size' states using linear regression.
    
    Args:
    - user_data: numpy array of shape (n, 6), where n is the number of timesteps,
                 and 6 represents the 6 DoF (x, y, z, yaw, pitch, roll).
    - window_size: int, the number of states to consider for the prediction
    
    Returns:
    - next_state: numpy array of shape (6,), representing the predicted next state.
    """
    if user_data.shape[0] < window_size:
        raise ValueError("Not enough data for prediction.")
    
    next_state = np.zeros(dof)
    time_steps = np.arange(window_size)
    
    # Perform linear regression on each DoF using the last 'window_size' states
    for i in range(dof):
        m, c = linear_regression(time_steps, user_data[-window_size:, i])
        next_state[i] = (m * (window_size+future_steps-1) + c +360)%360  # Predict the next state
    
    return next_state

# # Example usage
# # Assuming user_data is a numpy array with your 6DoF data for user1
# dof = 2
# user_data = np.random.rand(10, dof)  # Dummy data for demonstration

# next_state = predict_next_state_tlp(user_data, window_size=3,dof=dof)
# print("Predicted next state using the last 30 states:", next_state)
from sklearn.metrics import mean_squared_error
from math import sqrt   
def LR_prediction(window_size_lr=5,future_steps=10):
#     LR prediction
    
    # read ground truth data
    file_path = "./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv"
    trajectory_positions, trajectory_orientations = parse_trajectory_data(file_path,user_index='P01_V1')
    trajectory_positions.shape

    begin_frame_index = 0
    end_frame_index = 149
    # window_size_lr = 30
    # future_steps = 60
    dof = 3
    predicted_trajectory_positions = np.zeros(trajectory_positions[begin_frame_index:end_frame_index+1,:].shape)
    predicted_trajectory_orientations = np.zeros(trajectory_orientations[begin_frame_index:end_frame_index+1,:].shape)
    for frame_index in range(begin_frame_index+window_size_lr,end_frame_index+1 -future_steps +1):
        future_state = predict_next_state_tlp(trajectory_positions[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=future_steps)
        # print(trajectory_positions[frame_index-window_size_lr:frame_index,:])
        # print(next_state)

        predicted_trajectory_positions[frame_index+future_steps -1] = future_state
        # import pdb;pdb.set_trace()

        future_state = predict_next_state_tlp_rad(trajectory_orientations[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=1)
        # print(trajectory_positions[frame_index-window_size_lr:frame_index,:])
        # print(next_state)
        predicted_trajectory_orientations[frame_index+future_steps -1] = future_state
        
    file_path = "./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv"
    pred_file_path = "./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav_pred"+str(window_size_lr)+str(future_steps)+".csv"
    pred_difference_path = "./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav_pred_difference"+str(window_size_lr)+str(future_steps)+".csv"
    
    
    # read gt
    gt_df = pd.read_csv(file_path)
    df = gt_df.head(end_frame_index+1)
    # df
    # wirte into new predicted trajectory file
    df.iloc[0:end_frame_index+1,1:4] = predicted_trajectory_positions
    df.iloc[0:end_frame_index+1,4:7] = predicted_trajectory_orientations
    df_pred = df
    # df_pred
    df_pred.to_csv(pred_file_path,index=False)
    # get FoV and calculate loss
    # %run node_feature_utils.py
    cell2node,node2cell = cell2node2cell()
    trajectory_positions_pred, trajectory_orientations_pred = parse_trajectory_data(pred_file_path,user_index='P01_V1')
    ## get all cell occupied 
    occupied_cell_pred = get_all_cell_occupied(frame_index_off_set = 1051,
                                      frame_index_begin = 0,frame_index_end = end_frame_index)
    # len(occupied_cell_pred)
    # occupied_cell
    ## get all cell in the view frustum
    FoV_cell_pred = get_cell_in_view_frustum(trajectory_positions_pred,trajectory_orientations_pred, occupied_cell_pred)
    len(FoV_cell_pred)
    ## get feature for all cell
    # %run node_feature_utils.py
    node_occlusion_feature_df_pred = get_occlusion_feature_for_all_cell(FoV_cell_pred,occupied_cell_pred,
                                                               trajectory_positions_pred,
                                                               trajectory_orientations_pred,
                                                               trajectory_frame_index_begin=window_size_lr+future_steps-1,
                                                               trajectory_frame_index_end=end_frame_index)
    # load ground truth trajetory and get ground truth for node features
    file_path = "./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv"
    trajectory_positions, trajectory_orientations = parse_trajectory_data(file_path,user_index='P01_V1')
    trajectory_positions = trajectory_positions[0:end_frame_index+1,:]
    trajectory_orientations = trajectory_orientations[0:end_frame_index+1,:]
    occupied_cell = get_all_cell_occupied(frame_index_off_set = 1051,
                                          frame_index_begin = 0,frame_index_end = end_frame_index)
    # occupied_cell
    # %run node_feature_utils.py
    FoV_cell = get_cell_in_view_frustum(trajectory_positions,trajectory_orientations, occupied_cell)
    # %run node_feature_utils.py
    node_occlusion_feature_df = get_occlusion_feature_for_all_cell(FoV_cell,occupied_cell,trajectory_positions,trajectory_orientations,
                                  trajectory_frame_index_begin=window_size_lr+future_steps-1,trajectory_frame_index_end=end_frame_index)
    
    # Merging DataFrames on 'u', 'i', 'ts'
    joined_df = pd.merge( node_occlusion_feature_df, node_occlusion_feature_df_pred, on=['cell', 'ts'], suffixes=('_gt','_pred'))
    # Fill NaN values in label_pred with 4
    # joined_df['label_pred'].fillna(4, inplace=True)
    # joined_df

    # Calculating MSE between the 'label' columns
    mse = mean_squared_error(joined_df['label_pred'], joined_df['label_gt'])
    rmse = sqrt(mse)
    print('rmse',rmse,'window size',window_size_lr,'future steps',future_steps)
    joined_df['label'] = joined_df.label_pred-joined_df.label_gt
    cell_occlusion_level_distribution(joined_df,label='label')
    joined_df.to_csv(pred_difference_path,index=False)
    return 
