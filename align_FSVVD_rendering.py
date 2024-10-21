from ast import main
import pandas as pd
from FSVVD_data_utils import *
import open3d as o3d
import os
from point_cloud_FoV_utils import *
import gc
# fix alpha data error(remove alpha data)
# one a headless environment by enabling offscreen rendering and setting up a virtual display:

# Xvfb :99 -screen 0 800x600x24 & export DISPLAY=:99
# export DISPLAY=:99

# Xvfb :100 -screen 0 1920x1080x24 & export DISPLAY=:100


# given video name, user name, read resampled user behavior trajectory, read FSVVD, render from given FoV, save to file
# input is the video name, user name, output is the rendered video

    # video_name = 'Chatting'
    # user_name = 'HKY'
def save_fov_images_from_user_behaviror(video_name,user_name):
    ub_file_name = f'{user_name}_{video_name}_resampled.txt'
    # ub_file_name = 'RenZhichen_chatting_resampled.txt'
    # ub_file_name = 'GuoYushan_chatting_resampled.txt'
    # ub_file_name = 'liuxuya_chatting_resampled.txt'
    # ub_file_name = 'LHJ_chatting_resampled.txt'
    # ub_file_name = 'ChenYongting_chatting_resampled.txt'
    resampled_user_behavior_file_path = f'../point_cloud_data/processed_FSVVD/Resample_UB/{video_name}/'
    FSVVD_file_path = f'../point_cloud_data/processed_FSVVD/FSVVD_300/{video_name}/Raw/'
    ub_df = pd.read_csv(resampled_user_behavior_file_path + ub_file_name, delim_whitespace=True)
    FSVVD_ub_video_path = '../point_cloud_data/processed_FSVVD/ub_video/'
    # ub_df
    render_flag = False
    save = True
    image_width, image_height = np.array([1920, 1080])
    print('total frames',len(ub_df))

    # positions = [0,0,0]
    # orientations = [0,0,0]


    for frame in range(0,len(ub_df)):
    # for frame in range(0,300,60):
        ub_row = ub_df.iloc[frame]
        selected_position = ub_row[['HeadX', 'HeadY', 'HeadZ']].values
        selected_orientation = ub_row[['HeadRX', 'HeadRY', 'HeadRZ']].values
        # print('selected_position:',selected_position)
        # print('selected_orientation:',selected_orientation)

        pcd = o3d.io.read_point_cloud(FSVVD_file_path + f'{frame%300}_binary.ply')
        # print(f'{frame%300}_binary.ply')
        # o3d.visualization.draw_geometries([pcd])
        save_rendering_from_given_FoV_traces_fsvvd(pcd,selected_position,selected_orientation,frame,
                                                save=save,render_flag=render_flag,
                                                save_path=FSVVD_ub_video_path+ub_file_name.split('.')[0]+'/',
                                                save_filename=str(frame).zfill(4)+'.png')

# 
if __name__ == '__main__':
    video_name = 'Chatting'
    video_name = 'Pulling_trolley'
    video_name = 'Cleaning_whiteboard'
    video_name = 'News_interviewing'
    # video_name = 'Presenting'
    video_name = 'Sweep'
    # user_name = 'HKY'
    # full_user_list_chatting = ['HKY','LHJ','Guozhaonian','RenHongyu','Sunqiran','sulehan','LiaoJunjian','LHJ','TuYuzhao','yuchen','FengXuanqi','fupingyu','RenZhichen','WangYan','huangrenyi','ChenYongting','GuoYushan','liuxuya']
    Pulling_trolley_user =      ['TuYuzhao', 'Guozhaonian', 'fupingyu', 'FengXuanqi', 'WangYan', 'Sunqiran', 'LHJ', 'GuoYushan', 'ChenYongting', 'huangrenyi', 'sulehan', 'liuxuya', 'yuchen', 'LiaoJunjian', 'RenHongyu', 'RenZhichen', 'HKY']
    Cleaning_whiteboard_user =   ['RenHongyu', 'liuxuya', 'sulehan', 'GuoYushan', 'LHJ', 'RenZhichen', 'Guozhaonian', 'Sunqiran', 'fupingyu', 'yuchen', 'huangrenyi',  'WangYan', 'ChenYongting','HKY']
    Sweep_user = ['sulehan', 'LHJ', 'TuYuzhao', 'Sunqiran', 'yuchen', 'FengXuanqi', 'WangYan', 'huangrenyi', 'ChenYongting', 'LiaoJunjian', 'liuxuya', 'RenZhichen', 'Renhongyu', 'Guozhaonian', 'fupingyu', 'GuoYushan','HKY']
    Presenting_user = ['HKY', 'fupingyu', 'sulehan', 'yuchen', 'ChenYongting', 'WangYan', 'Sunqiran', 'GuoYushan', 'RenZhichen', 'liuxuya', 'huangrenyi', 'Guozhaonian']
    News_interviewing_user = ['HKY', 'Guozhaonian', 'liuxuya', 'fupingyu', 'RenHongyu', 'sulehan', 'RenZhichen', 'huangrenyi', 'LiaoJunjian', 'GuoYushan', 'Sunqiran', 'ChenYongting', 'yuchen', 'WangYan']
    # for user_name in ['HKY']:
    for user_name in Sweep_user:
        print(f'video_name:{video_name},user_name:{user_name}')
        save_fov_images_from_user_behaviror(video_name,user_name)
        gc.collect()
