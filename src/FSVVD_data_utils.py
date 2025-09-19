import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import os
from point_cloud_FoV_utils import *


def resample_dataframe(df, freq_hz):
    """
    Resample the dataframe based on the Timer column to a specified frequency using linear interpolation.
    Excludes the original 'Frame' column and assigns a new 'Frame' starting from 0.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe with at least a 'Timer' column.
    - freq_hz (float): Desired resampling frequency in Hertz (samples per second).
    
    Returns:
    - pd.DataFrame: Resampled dataframe with regular Timer intervals and a new 'Frame' column.
    """
    # Ensure the Timer column exists
    if 'Timer' not in df.columns:
        raise ValueError("Dataframe must contain a 'Timer' column.")
    
    # Sort the dataframe by Timer
    df_sorted = df.sort_values('Timer').reset_index(drop=True)
    
    # Set Timer as the index
    df_sorted.set_index('Timer', inplace=True)
    
    # Determine the sampling interval in seconds
    sample_interval = 1.0 / freq_hz
    
    # Create a new Timer index with the desired frequency
    timer_min = df_sorted.index.min()
    timer_max = df_sorted.index.max()
    
    # Use np.arange to create the new Timer index
    new_timer = np.arange(timer_min, timer_max, sample_interval)
    
    # Reindex the dataframe to include the new Timer points
    df_reindexed = df_sorted.reindex(df_sorted.index.union(new_timer)).sort_index()
    
    # Identify numerical columns to interpolate (excluding 'Frame' if it exists)
    numerical_cols = df_reindexed.columns.drop('#Frame', errors='ignore')
    
    # Interpolate numerical columns linearly
    df_reindexed[numerical_cols] = df_reindexed[numerical_cols].interpolate(method='linear')
    
    # Select only the new Timer points
    df_resampled = df_reindexed.loc[new_timer]
    
    # Reset the index to turn Timer back into a column
    df_resampled = df_resampled.reset_index().rename(columns={'index': 'Timer'})
    
    # Assign a new Frame column starting from 0
    df_resampled['#Frame'] = np.arange(len(df_resampled))
    
    # Optional: Reorder columns to place 'Frame' first
    cols = ['#Frame'] + [col for col in df_resampled.columns if col != '#Frame']
    df_resampled = df_resampled[cols]
    
    return df_resampled

def visualize_comparison(df_original, df_resampled, column, 
                        original_color='blue', resampled_color='red', 
                        original_label='Original', resampled_label='Resampled', 
                        title_suffix='', xlabel='Timer (seconds)', ylabel=None,
                        figsize=(12, 6), scatter_size=3, line_width=2, alpha=0.6):
    """
    Visualize the comparison between original and resampled data for a specified column.
    
    Parameters:
    - df_original (pd.DataFrame): Original dataframe containing 'Timer' and the specified column.
    - df_resampled (pd.DataFrame): Resampled dataframe containing 'Timer' and the specified column.
    - column (str): The name of the column to visualize (e.g., 'HeadX', 'HeadY', 'HeadZ').
    - original_color (str): Color for the original data points. Default is 'blue'.
    - resampled_color (str): Color for the resampled data line and points. Default is 'red'.
    - original_label (str): Label for the original data in the legend. Default is 'Original'.
    - resampled_label (str): Label for the resampled data in the legend. Default is 'Resampled'.
    - title_suffix (str): Additional text to append to the plot title. Default is empty.
    - xlabel (str): Label for the x-axis. Default is 'Timer (seconds)'.
    - ylabel (str): Label for the y-axis. If None, defaults to the column name.
    - figsize (tuple): Size of the plot figure. Default is (12, 6).
    - scatter_size (int): Size of the scatter points. Default is 3.
    - line_width (int): Width of the resampled data line. Default is 2.
    - alpha (float): Transparency level for the resampled scatter points. Default is 0.6.
    
    Returns:
    - None: Displays the plot.
    """
    # Check if the specified column exists in both dataframes
    if column not in df_original.columns:
        raise ValueError(f"Column '{column}' not found in the original dataframe.")
    if column not in df_resampled.columns:
        raise ValueError(f"Column '{column}' not found in the resampled dataframe.")
    
    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    plt.figure(figsize=figsize,dpi=300)
    
    # Plot original data as scatter points
    plt.scatter(df_original['Timer'], df_original[column], 
                color=original_color, label=f'{original_label} {column}', 
                zorder=5, s=scatter_size)
    
    # # # Plot resampled data as a continuous line
    # plt.plot(df_resampled['Timer'], df_resampled[column], 
    #          color=resampled_color, label=f'{resampled_label} {column}', 
    #          linewidth=line_width)
    
    # Optional: Add scatter points to resampled data
    plt.scatter(df_resampled['Timer'], df_resampled[column], 
                color=resampled_color, s=scatter_size, alpha=alpha,label=f'{resampled_label} {column}')
    
    # Title and labels
    plot_title = f'Comparison of Original and Resampled {column} Data'
    if title_suffix:
        plot_title += f' {title_suffix}'
    plt.title(plot_title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel if ylabel else column, fontsize=14)
    
    # Legend
    plt.legend(fontsize=12)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_columns(df, columns):
    sns.set(style="whitegrid")
    
    plt.figure(dpi=150)
    for column in columns:
        plt.scatter(df['Timer'], df[column], label=column,s=0.5)
    plt.xlabel('Timer (seconds)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


import pandas as pd
import numpy as np

def convert_orientation_to_sin_cos(df, orientation_columns, angle_unit='degrees'):
    """
    Convert specified orientation angle columns in a dataframe to their sine and cosine components.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing orientation angle columns.
    - orientation_columns (list of str): List of column names representing orientation angles (e.g., ['HeadRX', 'HeadRY', 'HeadRZ']).
    - angle_unit (str): Unit of the angles in the dataframe. Options are 'degrees' or 'radians'. Default is 'degrees'.

    Returns:
    - pd.DataFrame: New dataframe with sine and cosine columns added for each specified orientation column.
    """
    # Create a copy to avoid modifying the original dataframe
    df_sin_cos = df.copy()

    for col in orientation_columns:
        if col not in df_sin_cos.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

        # Convert angles to radians if they are in degrees
        if angle_unit == 'degrees':
            radians = np.deg2rad(df_sin_cos[col])
        elif angle_unit == 'radians':
            radians = df_sin_cos[col]
        else:
            raise ValueError("angle_unit must be either 'degrees' or 'radians'.")

        # Compute sine and cosine
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"
        df_sin_cos[sin_col] = np.sin(radians)
        df_sin_cos[cos_col] = np.cos(radians)
    # drop orientation_columns
    df_sin_cos.drop(orientation_columns, axis=1, inplace=True)



    return df_sin_cos

def convert_sin_cos_to_orientation(df, orientation_columns, angle_unit='degrees'):
    """
    Convert sine and cosine component columns back to orientation angles.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing sine and cosine columns for orientation angles.
    - orientation_columns (list of str): List of original orientation column names (e.g., ['HeadRX', 'HeadRY', 'HeadRZ']).
    - angle_unit (str): Unit of the output angles. Options are 'degrees' or 'radians'. Default is 'degrees'.

    Returns:
    - pd.DataFrame: New dataframe with the original orientation angle columns added.
    """
    # Create a copy to avoid modifying the original dataframe
    df_angles = df.copy()

    for col in orientation_columns:
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"

        if sin_col not in df_angles.columns or cos_col not in df_angles.columns:
            raise ValueError(f"Sine or cosine column for '{col}' not found in the dataframe.")

        # Compute the angle using arctan2
        radians = np.arctan2(df_angles[sin_col], df_angles[cos_col])

        # Convert radians to degrees if needed
        if angle_unit == 'degrees':
            angle = np.rad2deg(radians)
            # Ensure angles are within [0, 360)
            angle = np.mod(angle, 360)
        elif angle_unit == 'radians':
            angle = radians
            # Ensure angles are within [0, 2π)
            angle = np.mod(angle, 2 * np.pi)
        else:
            raise ValueError("angle_unit must be either 'degrees' or 'radians'.")

        # Add the angle to the dataframe
        df_angles[col] = angle

        # Drop the sine and cosine columns
        df_angles.drop([sin_col, cos_col], axis=1, inplace=True)

    return df_angles

def get_camera_intrinsic_matrix_fsvvd(image_width, image_height):
    fx, fy = 525, 525 # Focal length
    cx, cy = image_width/2, image_height/2 # Principal point
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
def get_camera_extrinsic_matrix_from_yaw_pitch_roll_fsvvd(yaw_degree, pitch_degree, roll_degree, t):
    # from world coordinate to camera coordinate, R is 3*3, t is 3*1, 
    t = np.array(t).reshape(3,1)

    def rotation_matrix_x(theta):
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
    def rotation_matrix_y(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
    def rotation_matrix_z(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    # get the rotation matrix
    # pitch_degree, yaw_degree, roll_degree = 0, 0, 0
    # here we set a 180 degree offset for pitch, 
    # because the camera is looking to the negative z axix in the world coordinate
    pitch, yaw, roll = np.radians(pitch_degree), np.radians(yaw_degree), np.radians(roll_degree) + np.radians(180)
    R = rotation_matrix_x(pitch) @ rotation_matrix_y(yaw) @ rotation_matrix_z(roll)

    # get 4*4 extrinsic matrix from R and t
    extrinsic_matrix = np.hstack((R, -R @ t))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
    return extrinsic_matrix


def save_rendering_from_given_FoV_traces_fsvvd(pcd,
                                               selected_position,
                                               selected_orientation,
                                               frame,
                                               save=False,
                                               render_flag=False,
                                               save_path=None,
                                               save_filename=None):     
    
    # pcd = get_pcd_data(point_cloud_name='longdress', trajectory_index=frame%150)
    # import pdb; pdb.set_trace()
    print(f"Frame {frame}: Loaded {len(pcd.points)} points.")
    # o3d.visualization.draw_geometries([pcd])
    # get the first row of ub_df
    # print(ub_row)



    # print(selected_position)
    # print(selected_orientation)
    # image_width, image_height = np.array([800,600])
    image_width, image_height = np.array([1920, 1080])
    intrinsic_matrix = get_camera_intrinsic_matrix_fsvvd(image_width, image_height)
    # Define camera extrinsic parameters (example values for rotation and translation)
    # pitch_degree, yaw_degree, roll_degree = selected_orientation

    pitch_degree, yaw_degree, roll_degree = selected_orientation
    # pitch_degree, yaw_degree, roll_degree = [0,0,0]
    para_eye = selected_position
    extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll_fsvvd(yaw_degree, pitch_degree, roll_degree, para_eye)
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
    # o3d.visualization.draw([pcd,coordinate_frame],
    #                        intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
    #                        raw_mode=True,show_skybox=False)        
    # pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # print('points after downsample:',len(pcd.points))
    # print('total points:',len(pcd.points)) 
    # pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
    # print('points in FoV:',len(pcd.points))
    # pcd = hidden_point_removal_fsvvd(pcd,para_eye)
    # print('points after hidden point removal:',len(pcd.points))

    # import pdb; pdb.set_trace()
    # Setting up the visualizer
    vis = o3d.visualization.Visualizer()
    # vis.create_window(width=image_width, height=image_height)
    vis.create_window(visible=render_flag)
    
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=para_eye)
    vis.add_geometry(pcd)

    # Change point size
    # render_option = vis.get_render_option()
    # render_option.point_size = 10.0  # Set this to your desired point size
    # vis.add_geometry(coordinate_frame)
    # print("my customize extrincis matrix:")
    # print(extrinsic_matrix,selected_orientation,selected_position,intrinsic_matrix)
    view_ctl = vis.get_view_control()
    # import pdb; pdb.set_trace()
    cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
    cam_pose_ctl.intrinsic.height = image_height
    cam_pose_ctl.intrinsic.width = image_width
    cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix
    cam_pose_ctl.extrinsic = extrinsic_matrix
    view_ctl.convert_from_pinhole_camera_parameters(cam_pose_ctl, allow_arbitrary=True)
    view_ctl.change_field_of_view()
    # render
    vis.poll_events()
    vis.update_renderer()
    if render_flag:
        vis.run()
    # w
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vis.capture_screen_image(save_path + save_filename, do_render=True)# set do_render=True to render the image on virtual display
        # vis.capture_screen_image(save_path + save_filename, do_render=False)# set do_render=False on your local computer with display
        # pass
    # check path exist or not, if not create it

        # if not os.path.exists('../result/'+point_cloud_name+'/'+user):
        #     os.makedirs('../result/'+point_cloud_name+'/'+user)        
        # vis.capture_screen_image('../result/'+point_cloud_name+'/'+user+'/'+prefix+'fov_'+str(trajectory_index).zfill(3)+'.png', do_render=False) 
    # index should have 3 digits
    vis.destroy_window()

  