import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def change2num_points_from_percentage_TLR():
    # read data
    # column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    voxel_size=128
    # pcd_name_list = ['longdress','loot','redandblack','soldier']
    pcd_name_list = ['soldier']
    history = 90
    # future = 60
    for future in [10]:
        for pcd_name in pcd_name_list:
            for user_i in tqdm(range(1,15)):
                participant = 'P'+str(user_i).zfill(2)+'_V1'
                prefix = f'{pcd_name}_VS{voxel_size}_TLR'
                prefix_original = prefix+'_per'
                original_node_feature_path = f'./data/{prefix_original}/{participant}node_feature{history}{future}.csv'
                output_node_feature_path = f'./data/{prefix}/{participant}node_feature{history}{future}.csv'
                original_df = pd.read_csv(original_node_feature_path)
                # update occlusion_feature=occupancy_feature*occupancy_feature
                original_df['occlusion_feature'] = original_df['occupancy_feature']*original_df['occlusion_feature']
                # save to the new file
                # check directory exists
                if not os.path.exists(f'./data/{prefix}'):
                    os.makedirs(f'./data/{prefix}')
                original_df.to_csv(output_node_feature_path, index=False)
                # return

def change2num_points_from_percentage():
    # read data
    # column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    voxel_size=128
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
    for pcd_name in pcd_name_list:
        for user_i in tqdm(range(1,28)):
            participant = 'P'+str(user_i).zfill(2)+'_V1'
            prefix = f'{pcd_name}_VS{voxel_size}'
            prefix_original = prefix+'_per'
            original_node_feature_path = f'./data/{prefix_original}/{participant}node_feature.csv'
            output_node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
            original_df = pd.read_csv(original_node_feature_path)
            # update occlusion_feature=occupancy_feature*occupancy_feature
            original_df['occlusion_feature'] = original_df['occupancy_feature']*original_df['occlusion_feature']
            # save to the new file
            # check directory exists
            if not os.path.exists(f'./data/{prefix}'):
                os.makedirs(f'./data/{prefix}')
            original_df.to_csv(output_node_feature_path, index=False)

def change2_percentage_from_num_points():
    # read data
    # column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    voxel_size=64
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
    for pcd_name in pcd_name_list:
        for user_i in tqdm(range(1,28)):
            participant = 'P'+str(user_i).zfill(2)+'_V1'
            prefix = f'{pcd_name}_VS{voxel_size}_per'
            prefix_original = f'{pcd_name}_VS{voxel_size}'
            original_node_feature_path = f'./data/{prefix_original}/{participant}node_feature.csv'
            output_node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
            original_df = pd.read_csv(original_node_feature_path)
            # update occlusion_feature=occupancy_feature*occupancy_feature
            # original_df['occlusion_feature'] = original_df['occlusion_feature']/original_df['occupancy_feature']
            # original_df['occlusion_feature'] = 0 if original_df['occupancy_feature'] == 0 else original_df['occlusion_feature']/original_df['occupancy_feature']

            original_df['occlusion_feature'] = np.where(original_df['occupancy_feature'] == 0, 0, original_df['occlusion_feature'] / original_df['occupancy_feature'])
            # import pdb; pdb.set_trace()
            # save to the new file
            # check directory exists
            if not os.path.exists(f'./data/{prefix}'):
                os.makedirs(f'./data/{prefix}')
            original_df.to_csv(output_node_feature_path, index=False)

if __name__ == "__main__":
    # change2num_points_from_percentage_TLR()
    # change2num_points_from_percentage()
    change2_percentage_from_num_points()