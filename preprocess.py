# import open3d as o3d
import os
# from point_cloud_FoV_utils import get_pcd_data_original
from tqdm import tqdm
import pandas as pd

# def downsample_binary_pcd_data():
#     # Downsample original pcd and save to the binary pcd data
#     for point_cloud_name in ['longdress','loot','redandblack','soldier']:
#         if not os.path.exists(f'./data/{point_cloud_name}'):
#             os.makedirs(f'./data/{point_cloud_name}')
#         for trajectory_index in tqdm(range(0, 151)):
#             pcd = get_pcd_data_original(point_cloud_name, trajectory_index)
#             pcd = pcd.voxel_down_sample(voxel_size=8)
#             o3d.io.write_point_cloud(f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply', pcd, write_ascii=False)
#     return pcd

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


if __name__ == "__main__":
    # downsample_binary_pcd_data()
    change2num_points_from_percentage_TLR()