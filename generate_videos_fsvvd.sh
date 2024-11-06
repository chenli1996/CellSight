#!/bin/bash

# This script is used to generate videos from PNG files for multiple users and multiple video names.

# List of video names
# video_name_list=('Presenting','Chatting','Pulling_trolley','Cleaning_whiteboard','Sweep','News_interviewing')
# video_name_list=('Sweep','Cleaning_whiteboard')
# List of users
Chatting_user=('HKY' 'LHJ' 'Guozhaonian' 'RenHongyu' 'Sunqiran' 'sulehan' 'LiaoJunjian' 'LHJ' 'TuYuzhao' 'yuchen' 'FengXuanqi' 'fupingyu' 'RenZhichen' 'WangYan' 'huangrenyi' 'ChenYongting' 'GuoYushan' 'liuxuya')
Pulling_trolley_user=('TuYuzhao' 'Guozhaonian' 'fupingyu' 'FengXuanqi' 'WangYan' 'Sunqiran' 'LHJ' 'GuoYushan' 'ChenYongting' 'huangrenyi' 'sulehan' 'liuxuya' 'yuchen' 'LiaoJunjian' 'RenHongyu' 'RenZhichen' 'HKY')
Cleaning_whiteboard_user=('RenHongyu' 'liuxuya' 'sulehan' 'GuoYushan' 'LHJ' 'RenZhichen' 'Guozhaonian' 'Sunqiran' 'fupingyu' 'yuchen' 'huangrenyi' 'WangYan' 'ChenYongting' 'HKY')
Sweep_user=('sulehan' 'LHJ' 'TuYuzhao' 'Sunqiran' 'yuchen' 'FengXuanqi' 'WangYan' 'huangrenyi' 'ChenYongting' 'LiaoJunjian' 'liuxuya' 'RenZhichen' 'RenHongyu' 'Guozhaonian' 'fupingyu' 'GuoYushan' 'HKY')
Presenting_user=('HKY' 'fupingyu' 'sulehan' 'yuchen' 'ChenYongting' 'WangYan' 'Sunqiran' 'GuoYushan' 'RenZhichen' 'liuxuya' 'huangrenyi' 'Guozhaonian')
News_interviewing_user=('HKY' 'Guozhaonian' 'liuxuya' 'fupingyu' 'RenHongyu' 'sulehan' 'RenZhichen' 'huangrenyi' 'LiaoJunjian' 'GuoYushan' 'Sunqiran' 'ChenYongting' 'yuchen' 'WangYan')

user_list=("${Pulling_trolley_user[@]}")

video_name='Pulling_trolley'
echo "Video name: $video_name"


file_path="/scratch/cl5089/point_cloud_data/processed_FSVVD/ub_video/"

# Create the output directory if it doesn't exist
mkdir -p "${file_path}user_videos"

# Loop through each user
for user in "${user_list[@]}"
do
    image_folder="${file_path}${user}_${video_name}_resampled"
    video_output_path="${file_path}user_videos/${user}_${video_name}_video_60fps.mp4"
    
    # Check if the output video already exists
    if [ -f "$video_output_path" ]; then
        echo "Video for $user already exists. Skipping."
        continue
    fi

    # Check if the user's image folder exists
    if [ -d "$image_folder" ]; then
        echo "Processing $user for $video_name..."
        
        # Run ffmpeg command to generate the video
        ffmpeg -framerate 60 -pattern_type glob -i "${image_folder}/*.png" -c:v libx264 -pix_fmt yuv420p "$video_output_path"
    else
        echo "Folder $image_folder does not exist. Skipping $user for $video_name."
    fi
done

echo "Video generation completed!"