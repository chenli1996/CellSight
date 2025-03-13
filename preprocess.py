from webbrowser import get
import open3d as o3d
import os
from point_cloud_FoV_utils import get_pcd_data_FSVVD_filtered, get_pcd_data_original
from tqdm import tqdm
import pandas as pd

def downsample_binary_pcd_data():
    # Downsample original pcd and save to the binary pcd data
    for point_cloud_name in ['longdress','loot','redandblack','soldier']:
        if not os.path.exists(f'./data/{point_cloud_name}'):
            os.makedirs(f'./data/{point_cloud_name}')
        for trajectory_index in tqdm(range(0, 151)):
            pcd = get_pcd_data_original(point_cloud_name, trajectory_index)
            pcd = pcd.voxel_down_sample(voxel_size=8)
            o3d.io.write_point_cloud(f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply', pcd, write_ascii=False)
    return pcd

def get_pcd_data_FSVVD(point_cloud_name='Chatting', trajectory_index=0):
    FSVVD_file_path = f'../point_cloud_data/processed_FSVVD/FSVVD_300/{point_cloud_name}/Raw/'
    pcd = o3d.io.read_point_cloud(FSVVD_file_path + f'{trajectory_index%300}_binary.ply')
    return pcd

def downsample_binary_pcd_data_FSVVD():
    # Downsample original pcd and save to the binary pcd data
    for point_cloud_name in  ['Chatting','Pulling_trolley','News_interviewing','Sweep']:
        FSVVD_file_path_downsample = f'../point_cloud_data/processed_FSVVD/FSVVD_300_downsample/{point_cloud_name}/Filtered/'
        if not os.path.exists(FSVVD_file_path_downsample):
            os.makedirs(FSVVD_file_path_downsample)
        for trajectory_index in tqdm(range(0, 300)):
            pcd = get_pcd_data_FSVVD_filtered(point_cloud_name, trajectory_index)
            # pcd = get_pcd_data_FSVVD(point_cloud_name, trajectory_index)
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            o3d.io.write_point_cloud(f'{FSVVD_file_path_downsample}/{trajectory_index}_binary_downsampled.ply', pcd, write_ascii=False)
    return pcd

def binary_pcd_data():
    # Downsample original pcd and save to the binary pcd data
    for point_cloud_name in ['longdress','loot','redandblack','soldier']:
        if not os.path.exists(f'./data/binary_original/{point_cloud_name}'):
            os.makedirs(f'./data/binary_original/{point_cloud_name}')
        for trajectory_index in tqdm(range(0, 151)):
            pcd = get_pcd_data_original(point_cloud_name, trajectory_index)
            o3d.io.write_point_cloud(f'./data/binary_original/{point_cloud_name}/frame{trajectory_index}_binary.ply', pcd, write_ascii=False)
    return pcd



if __name__ == "__main__":
    binary_pcd_data()
    downsample_binary_pcd_data()
    downsample_binary_pcd_data_FSVVD()