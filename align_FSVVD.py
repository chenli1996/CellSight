import os
import open3d as o3d
import numpy as np
import pandas as pd
from FSVVD_data_utils import *
from tqdm import tqdm


video_name = 'chatting'
user_behavior_file_path = '../point_cloud_data/FSVVD/ACM_MM23 User Behavior Dataset with Tools/User Movement/'
files = os.listdir(user_behavior_file_path)
files_chatting = [file for file in files if video_name in file]
# files_chatting

## read user behavior trajectory
for file in tqdm(files_chatting):
    file_path = user_behavior_file_path + file
    # if file != 'GuoYushan_chatting.txt':
    #     continue
    # Read the first row to get the column names
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        column_names = first_line.split(', ')
    
    # Read the rest of the file using tabs as delimiters
    df_full = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    
    # Set the column names
    df_full.columns = column_names
    
    # Show the first few rows of the DataFrame
    df_full.head()
    # break

    numeric_cols = ['#Frame', 'Timer', 'HeadX', 'HeadY', 'HeadZ']
    # Orientation columns
    orientation_cols = ['HeadRX', 'HeadRY', 'HeadRZ','LEyeRX','LEyeRY', 'LEyeRZ','REyeRX', 'REyeRY', 'REyeRZ']
    sub_column_names = numeric_cols + orientation_cols
    df = df_full[sub_column_names]
    # df = df.head(10)
    df.head()


    # covert to sin cos df
    # Convert to sine and cosine
    df_sin_cos = convert_orientation_to_sin_cos(df, orientation_cols)
    df_sin_cos.head()

    # Resample to 60Hz
    resampled_df = resample_dataframe(df_sin_cos, 60)
    # print(df)
    resampled_df

    # convert back to angles
    resampled_df = convert_sin_cos_to_orientation(resampled_df, orientation_cols)
    resampled_df.head()

    # Save the resampled user behavior trajectory
    # round(resampled_df, 6).to_csv('chatting_3_resampled.csv', index=False)
    resampled_df = round(resampled_df, 6)
    resampled_df


    # save resampled_df to file
    resampled_user_behavior_file_path = '../point_cloud_data/processed_FSVVD/Resample_UB/Chatting/'
    if not os.path.exists(os.path.dirname(resampled_user_behavior_file_path)):
        os.makedirs(os.path.dirname(resampled_user_behavior_file_path))
    resampled_file_name = file.replace('.txt', '_resampled.txt')
    resampled_df.to_csv(resampled_user_behavior_file_path + resampled_file_name, index=False, sep=' ')
    # break
