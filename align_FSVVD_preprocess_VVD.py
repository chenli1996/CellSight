import numpy as np
import open3d as o3d
import os
import pandas as pd
from FSVVD_data_utils import resample_dataframe, visualize_comparison, convert_orientation_to_sin_cos,convert_sin_cos_to_orientation
# read FSVVD and preprocess to 300 frames


# fix alpha data error(remove alpha data)


def fix_ply_alpha(file_path, file_name, fixed_file_path):
    # Construct full input and output paths
    input_full_path = os.path.join(file_path, file_name)
    output_file_name = file_name.replace(".ply", "_fixed.ply")
    output_full_path = os.path.join(fixed_file_path, output_file_name)

    # Read the .ply file
    with open(input_full_path, 'r') as file:
        lines = file.readlines()

    # Detect the end of the header
    header_end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end_idx = i
            break

    if header_end_idx is None:
        raise ValueError("Invalid .ply file: Missing end_header")

    # Process and modify the header
    original_header = lines[:header_end_idx + 1]
    new_header = []
    for line in original_header:
        if not line.strip().startswith("property uchar alpha"):
            new_header.append(line)
    # Alternatively, using list comprehension:
    # new_header = [line for line in original_header if not line.strip().startswith("property uchar alpha")]

    # Process vertex data by removing the last element (alpha)
    vertex_data = lines[header_end_idx + 1:]
    fixed_data = []
    for line in vertex_data:
        elements = line.strip().split()
        if len(elements) == 7:  # x, y, z, r, g, b, alpha
            fixed_line = " ".join(elements[:6])  # Remove the alpha value
            fixed_data.append(fixed_line)
        else:
            # Handle lines that do not have 7 elements (if any)
            fixed_data.append(line.strip())

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_full_path), exist_ok=True)

    # Write the modified header and vertex data to the new file
    with open(output_full_path, 'w') as file:
        file.writelines(new_header)
        file.write("\n")  # Ensure there's a newline after the header
        file.write("\n".join(fixed_data))
        file.write("\n")  # Optional: Add a newline at the end of the file

    print(f"Fixed .ply file saved as {output_full_path}")
    return output_full_path

# Example Usage
# file_path = '../../Chatting/Raw/'
# file_name = 'chatting_3_raw.ply'
# fixed_file_path = '../../Chatting/Fixed/'
# fix_ply_alpha(file_path, file_name, fixed_file_path)





video_name = 'Chatting'

raw_file_path = f'../point_cloud_data/FSVVD/{video_name}/Raw/'
# raw_file_path = '../point_cloud_data/FSVVD/Pulling_trolley/Raw/'
fixed_file_path = f'../point_cloud_data/processed_FSVVD/fixed_alpha/{video_name}/Raw/'
if not os.path.exists(fixed_file_path):
    os.makedirs(fixed_file_path)

# get all files in the directory raw_file_path

files = os.listdir(raw_file_path)
for file in files:
    if file.endswith('.ply'):
        # only process file with name '_number_' and number is less than 300
        if int(file.split('_')[-2]) >= 300: # do not process file with number >= 300
        # if int(file.split('_')[-2]) >= 1: # do not process file with number >= 300
            continue
        fixed_file = fix_ply_alpha(raw_file_path, file, fixed_file_path)
        # Load the fixed .ply file
        pcd = o3d.io.read_point_cloud(fixed_file)

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        # w

# read FSVVD and preprocess to 300 frames to binary ply

input_file_path = fixed_file_path
# input_file_path  = '../point_cloud_data/FSVVD/Chatting/Raw/'
output_file_path = f'../point_cloud_data/processed_FSVVD/FSVVD_300/{video_name}/Raw/'
# read all ply files from input_file_path and save to output_file_path with write_ascii=False using open3d
files = os.listdir(input_file_path)
# remove .DS_Store file if any
files = [file for file in files if file != '.DS_Store']
files.sort(key=lambda x: int(x.split('_')[-3]))
# files

if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))

for frame_index in range(0,300):
    # frame_index = 29
    selected_file = files[frame_index%len(files)]
    # print(selected_file)
    pcd = o3d.io.read_point_cloud(input_file_path+selected_file)
    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    
    o3d.io.write_point_cloud(f'{output_file_path}{frame_index}_binary.ply', pcd, write_ascii=False)
    # w


