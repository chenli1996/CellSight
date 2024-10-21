from re import T
import numpy as np
import open3d as o3d
from node_feature_utils import parse_trajectory_data
import os
from itertools import chain
from point_cloud_FoV_utils import *
from tqdm import tqdm

def save_rendering_from_given_FoV_traces(trajectory_positions,trajectory_orientations,
                                trajectory_index,point_cloud_name='longdress',user='P03_V1',prefix='',save=False,render_flag=False):    
    # pcd = get_pcd_data(point_cloud_name=point_cloud_name, trajectory_index=trajectory_index%150)
    # pcd = get_pcd_data_binary(point_cloud_name=point_cloud_name, trajectory_index=trajectory_index%150)
    # point_cloud_path = f'../point_cloud_data/tongyu/frame0_binary.ply'    
    # pcd_ty = o3d.io.read_point_cloud(point_cloud_path)
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,149]))#longdress

    # point_cloud_path = f'../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
    # pcd = o3d.io.read_point_cloud(point_cloud_path)

    pcd = get_pcd_data_original(point_cloud_name=point_cloud_name, trajectory_index=trajectory_index%150)
    # pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=trajectory_index%150)

    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,149]))#longdress
    # get XYZ data
    selected_position = trajectory_positions[trajectory_index]
    # para_eye = [i*1024/1.8 for i in selected_position]
    para_eye = [i*1024/1.8 for i in selected_position]
    para_eye[2] = -para_eye[2]
    para_eye = np.array(para_eye).reshape(3,1)
    # get yaw pitch roll and to orientation
    selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
    pitch_degree, yaw_degree, roll_degree = selected_orientation  # Convert degrees to radians if necessary
    # pitch_degree, yaw_degree, roll_degree = [0,-85,0]
    # image_width, image_height = np.array([1280, 720])
    image_width, image_height = np.array([1920, 1080])
    intrinsic_matrix = get_camera_intrinsic_matrix(image_width, image_height)
    # Define camera extrinsic parameters (example values for rotation and translation)
    extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, para_eye)
    # downsample and remove hidden points

    if prefix == 'visible_points':
        pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)
        # pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=8)
        pcd = hidden_point_removal(pcd,para_eye)
        # pcd = hidden_point_removal(pcd,[100,300,500])


    # Setting up the visualizer
    vis = o3d.visualization.Visualizer()
    # vis.create_window(width=image_width, height=image_height)
    vis.create_window(visible=render_flag)
    
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=para_eye)
    vis.add_geometry(pcd)
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
    # check path exist or not, if not create it
        output_file_path = './data/8i_user_videos/'+point_cloud_name+'/'+user+'/'
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)      
        vis.capture_screen_image(output_file_path+prefix+'fov_'+str(trajectory_index).zfill(3)+'.png', do_render=True) 
    # index should have 3 digits
    vis.destroy_window()



pcd_name='longdress'
participant='P01_V1'
positions,orientations = get_point_cloud_user_trajectory(pcd_name='longdress',participant=participant)
end_index = len(positions)
print('end_index:',end_index)
# end_index = 1
for index in tqdm(range(0, end_index,1)):
    print('index:',index)
    save_rendering_from_given_FoV_traces(positions,orientations,
                            trajectory_index=index,point_cloud_name=pcd_name,user=participant,render_flag=False,save=True,
                            prefix='visible_points'
                            )
       