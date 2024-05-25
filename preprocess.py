import open3d as o3d
import os
from point_cloud_FoV_utils import get_pcd_data_original
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




if __name__ == "__main__":
    downsample_binary_pcd_data()