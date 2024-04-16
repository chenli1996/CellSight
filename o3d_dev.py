import warnings
import pdb
import os
import numpy as np
import open3d as o3d

# Load the point cloud data
# pcd = o3d.io.read_point_cloud("./8i/longdress/longdress/Ply/longdress_vox10_1051.ply")
longdress_path = '/Users/chenli/research/point cloud/8i/longdress/longdress/Ply/longdress_vox10_1051.ply'
loot_path = '/Users/chenli/research/point cloud/8i/loot/loot/Ply/loot_vox10_1000.ply'
redandblack_path = '/Users/chenli/research/point cloud/8i/redandblack/redandblack/Ply/redandblack_vox10_1450.ply'
soldier_path = '/Users/chenli/research/point cloud/8i/soldier/soldier/Ply/soldier_vox10_0536.ply'
pcd = o3d.io.read_point_cloud(longdress_path)

# # Load the point cloud data
# point_cloud = o3d.io.read_point_cloud("./8i/longdress/longdress/Ply/longdress_vox10_1097.ply")

# # Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])

# # show point cloud on screen
# o3d.visualization.draw_geometries(
#     [pcd],
#     zoom=0.7,
#     front=[0., 0., 1],
#     # lookat=[2.6172, 2.0475, 1.532],
#     lookat=[400, 400, 0],
#     # eye=[100,0,0],
#     up=[0., 1, 0.]
# )
# vis = o3d.visualization.Visualizer()
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

# Function to convert Euler angles to direction vector (simplified)
def euler_to_direction(yaw, pitch, roll):
    # Assuming yaw (Z), pitch (Y), roll (X) in radians for simplicity
    direction = np.array([
        -np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
        
    ])
    return direction

trajectory_positions, trajectory_orientations = parse_trajectory_data("./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv")

# import pdb;pdb.set_trace()


trajectory_index = 0#2534


# get XYZ data
selected_position = trajectory_positions[trajectory_index]
para_eye = [i*1000/1.75 for i in selected_position]
print(para_eye)



# get yaw pitch roll and to orientation
selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary
# Calculate the direction vector from the orientation
orientation = euler_to_direction(yaw, pitch, roll)
print('orientation',orientation)

# para_eye = [200,500,1000]
# para_eye = [200,1024,1000]
# orientation = [0,0,-1]
para_lookat = [para_eye[i]+orientation[i] for i in range(3)]
# print(para_lookat)




o3d.visualization.draw([pcd],lookat=para_lookat,eye=para_eye,up=[0., 1, 0.],field_of_view=5)

# o3d.visualization.draw_geometries(
#     [pcd],
#     zoom=1,
#     front=[0., -0., 1],
#     # lookat=[2.6172, 2.0475, 1.532],
#     lookat=[400, 400, 0],
#     up=[0., 1, 0.])


# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the geometry to the Visualizer
# vis.add_geometry(pcd)


# ctr = vis.get_view_control()
# ctr.set_lookat([0, 400, 0])

# # camera location
# # the top head direction of the camera
# ctr.set_up([-np.power(0.5, 0.5), np.power(0.5, 0.5), 0.])
# ctr.set_up([0, 1, 0.])


# # Optionally, adjust view parameters here

# # Run the visualizer to display the window
# # You might want to automate or trigger the next step programmatically depending on your use case
# vis.run()

# # Save the screen image
# vis.capture_screen_image("your_image_filename.png")

# # Destroy the visualizer window
# vis.destroy_window()






# vis.draw([pcd],lookat=[400,400,0],eye=[100,0,0],up=[0., 1, 0.])

# import open3d as o3d
# import numpy as np



# # Define 6DoF coordinates (example values)
# x, y, z = 400.0, 10000.0, 0.0  # Position
# roll, pitch, yaw = np.radians(10), np.radians(20), np.radians(30)  # Orientation converted to radians

# # Convert orientation (roll, pitch, yaw) to rotation matrix
# R = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))

# # Camera parameters (assuming Open3D version supports it)
# camera_params = {
#     "position": (x, y, z),
#     "lookat": (0, 0, 0),  # Modify as needed
#     "up": (0, 1, 0),  # Typically the Y-axis is up
#     "field_of_view": 60  # Example FOV, adjust as necessary
# }

# # Visualization with custom camera settings
# o3d.visualization.draw({
#     "name": "Custom Visualization",
#     "geometry": pcd,
#     "camera_position": camera_params["position"],
#     "camera_lookat": camera_params["lookat"],
#     "camera_up": camera_params["up"],
#     "field_of_view": camera_params["field_of_view"]
    
# })




# o3d.visualization.draw([pcd],field_of_view=90,eye=[1000,0,0],up=[1,0,0])


# FOV_STEP = 30
# # non-blocking visualization
# vis = o3d.visualization.Visualizer()
# # cam = o3d.visualization.rendering.Camera()
# vis.create_window(
    # visible=True
# )
# # import pdb;pdb.set_trace()
# vis.add_geometry(pcd)
# vis.update_geometry(pcd)
# opt = vis.get_render_option()
# opt.show_coordinate_frame = True
# ctr = vis.get_view_control()
# # ctr.change_field_of_view(step=FOV_STEP)
# # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
# # ctr.set_zoom(3) # less -> larger, adjust distance
# ## set up point cloud obj params
# # front direction of point cloud obj
# # ctr.set_lookat([200, 600, 400])
# # ctr.set_front([0., 0., 1])
# # camera location
# # the top head direction of the camera
# # ctr.set_up([-np.power(0.5, 0.5), np.power(0.5, 0.5), 0.])
# # ctr.set_up([0, 1, 0.])
# # ctr.set_zoom(0.01)
# # fov = o3d.visualization.rendering.Camera.get_field_of_view()
# # print(fov)

# # param = ctr.convert_to_pinhole_camera_parameters()
# # o3d.io.write_pinhole_camera_parameters('open3d_test_results/camera_params.json', param)

# # # set up camera params
# # param = o3d.io.read_pinhole_camera_parameters('open3d_test_results/camera_params.json')
# # ctr.convert_from_pinhole_camera_parameters(param)


# # vis.poll_events()
# # vis.update_geometry(pcd)
# # vis.update_renderer()
# vis.run()
# # vis.capture_screen_image('./open3d_test_results/' + DIRECTORY[-4:] + '.png',do_render=True)
# vis.destroy_window()





# def custom_draw_geometry(pcd):
#     # The following code achieves the same effect as:
#     # o3d.visualization.draw_geometries([pcd])
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.run()
#     vis.destroy_window()
    
# custom_draw_geometry(point_cloud)

# def custom_draw_geometry_load_option(pcd):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.get_render_option().load_from_json("./tile_data/open3d_test_results/camera_params.json")
#     vis.run()
#     vis.destroy_window()
    
# custom_draw_geometry_load_option(point_cloud)    



# def custom_draw_geometry_with_custom_fov(pcd, fov_step):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     ctr = vis.get_view_control()
#     print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
#     ctr.change_field_of_view(step=fov_step)
#     print(fov_step)
#     # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
#     vis.run()
#     vis.destroy_window()
    
# custom_draw_geometry_with_custom_fov(point_cloud, -20)

# def custom_draw_geometry_with_rotation(point_cloud):

#     def rotate_view(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(10.0, 0.0)
#         return False

#     o3d.visualization.draw_geometries_with_animation_callback([point_cloud],
#                                                               rotate_view)

# custom_draw_geometry_with_rotation(point_cloud)    