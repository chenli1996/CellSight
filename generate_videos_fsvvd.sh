#!/bin/bash
# this script is used to generate videos from PNG files for multiple users
# List of users
video_name="Presenting"
Presenting_user=('HKY' 'fupingyu' 'sulehan' 'yuchen' 'ChenYongting' 'WangYan' 'Sunqiran' 'GuoYushan' 'RenZhichen' 'liuxuya' 'huangrenyi' 'Guozhaonian')
# Presenting_user=('HKY')


file_path="/scratch/cl5089/point_cloud_data/processed_FSVVD/ub_video/"
# Create the output directory if it doesn't exist
mkdir -p user_videos

# Loop through each user
for user in "${Presenting_user[@]}"
do
    image_folder="${file_path}${user}_Presenting_resampled"
    video_output_path="${file_path}user_videos/${user}_${video_name}_video_60fps.mp4"
    
    # Check if the user's image folder exists
    if [ -d "$image_folder" ]; then
        echo "Processing $user..."
        
        # Run ffmpeg command to generate the video
        ffmpeg -framerate 60 -pattern_type glob -i "${image_folder}/*.png" -c:v libx264 -pix_fmt yuv420p "$video_output_path"
    else
        echo "Folder $image_folder does not exist. Skipping $user."
    fi
done

echo "Video generation completed!"