import open3d as o3d

# # Load the point cloud data
point_cloud = o3d.io.read_point_cloud("../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1097.ply")

# # Visualize the point cloud
# # o3d.visualization.draw_geometries([point_cloud])




import csv

def parse_trajectory_data(file_path):
    positions = []
    orientations = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        for row in csvreader:
            # Assuming the CSV columns are ordered as mentioned
            positions.append([float(row[1]), float(row[2]), float(row[3])])
            orientations.append([float(row[4]), float(row[5]), float(row[6])])
    return positions, orientations

trajectory_positions, trajectory_orientations = parse_trajectory_data("../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv")



import numpy as np

def create_trajectory_lines(positions):
    # Create points from positions
    points = np.array(positions)
    # Create lines based on consecutive points
    lines = [[i, i+1] for i in range(len(points)-1)]
    # Create colors for the lines
    colors = [[1, 0, 0] for i in range(len(lines))]  # Red lines
    # Create a line set object in Open3D
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

trajectory_line_set = create_trajectory_lines(trajectory_positions)



# # Visualize the point cloud along with the trajectory
# o3d.visualization.draw_geometries([point_cloud, trajectory_line_set])

import open3d as o3d

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1097.ply")

# Function to convert Euler angles to direction vector (simplified)
def euler_to_direction(yaw, pitch, roll):
    # Assuming yaw (Z), pitch (Y), roll (X) in radians for simplicity
    direction = np.array([
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
        -np.cos(yaw) * np.cos(pitch),
        
    ])
    return direction

# Select a specific position and orientation from your data
selected_position = np.array(trajectory_positions[0])  # For example, the first position
selected_position = [-1000,1000,1000]
selected_orientation = np.array(trajectory_orientations[0])  # First orientation (yaw, pitch, roll)
pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary

# Calculate the direction vector from the orientation
direction = euler_to_direction(yaw, pitch, roll)
# import pdb;pdb.set_trace()
# Set up the visualization window and camera
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)

# Calculate the "look-at" parameters
eye = selected_position  # Camera position
center = selected_position + direction  # A point directly in front of the camera
up = np.array([0, 1, 0])  # Assuming Y-up coordinate system

# Apply the "look-at" parameters to the view control
# view_control = vis.get_view_control()
# view_control.set_lookat(center)
# view_control.set_front(center - eye)
# view_control.set_up(up)
# view_control.set_zoom(1)  # Adjust zoom if necessary



import open3d
vis = open3d.visualization.Visualizer()
vis.create_window() # the 0.17.0 version demands create_window() first, otherwise gives segmentation fault. Why?
ctr = vis.get_view_control() 
assert id(ctr) == id(vis.get_view_control())  # assertion error.

# Run the visualization
# vis.run()
# vis.destroy_window()
