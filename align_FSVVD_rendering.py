import pandas as pd
from FSVVD_data_utils import *
import open3d as o3d
import os
from point_cloud_FoV_utils import *
import gc


video_name = 'Chatting'
user_name = 'ChenYongting'

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
render_flag = True
save = False
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
                                               save=True,render_flag=False,
                                               save_path=FSVVD_ub_video_path+ub_file_name.split('.')[0]+'/',
                                               save_filename=str(frame).zfill(4)+'.png')
