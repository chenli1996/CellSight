import numpy as np
import open3d as o3d
from node_feature_utils import parse_trajectory_data
import os
from itertools import chain
# from test_draw import draw_my
# user a fixed random seed
np.random.seed(0)
def get_camera_intrinsic_matrix(image_width, image_height):
    fx, fy = 525, 525 # Focal length
    cx, cy = image_width/2, image_height/2 # Principal point
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
def get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, t):
    # from world coordinate to camera coordinate, R is 3*3, t is 3*1, 
    t = np.array(t).reshape(3,1)
    # t is camera position in world coordinate(numpy array)     
    # Define camera extrinsic parameters (example values for rotation and translation)
    # define x,y,z rotation matrix
    # here we use left-hand coordinate system
    def rotation_matrix_x(theta):
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), np.sin(theta)],
                        [0, -np.sin(theta), np.cos(theta)]])
    def rotation_matrix_y(theta):
        return np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]])
    def rotation_matrix_z(theta):
        return np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    # get the rotation matrix
    # pitch_degree, yaw_degree, roll_degree = 0, 0, 0
    # here we set a 180 degree offset for pitch, 
    # because the camera is looking to the negative z axix in the world coordinate
    pitch, yaw, roll = np.radians(-pitch_degree)+np.radians(180), np.radians(-yaw_degree), np.radians(-roll_degree)
    R = rotation_matrix_x(pitch) @ rotation_matrix_y(yaw) @ rotation_matrix_z(roll)

    # R = np.eye(3) # Identity matrix for rotation
    # t = np.array([[200], [500], [1000]]) # Translation
    # get 4*4 extrinsic matrix from R and t
    extrinsic_matrix = np.hstack((R, -R @ t))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
    return extrinsic_matrix

def get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height):
    far_near_plane = np.array([10, 10000])
    # Transform point cloud to camera coordinate system
    points_homogeneous = np.hstack((np.asarray(pcd.points), np.ones((len(pcd.points), 1)))) # shape is (n, 4)
    camera_coord_points = extrinsic_matrix @ points_homogeneous.T # shape is (4,4) * (4, n) = (4, n)
    camera_coord_points = camera_coord_points[:3, :] # shape is (3, n)
    # Project points onto the image plane
    projected_points = intrinsic_matrix @ camera_coord_points
    # # Normalize by the third (z) component, only on x,y, not on z, so that we can use projected_points[2, :] to get the far/near plane
    projected_points[0:2,:] /= projected_points[2, :] # Normalize by the third (z) component
    # Filter points based on image dimensions (example dimensions)
    in_fov_indices = np.where((projected_points[0, :] >= 0) & (projected_points[0, :] < image_width) & 
                            (projected_points[1, :] >= 0) & (projected_points[1, :] < image_height)  
                                # )
                            & (projected_points[2, :] > far_near_plane[0]) & 
                            (projected_points[2, :] < far_near_plane[1]))
    filtered_points = np.array(pcd.points)[in_fov_indices]
    # Create a new point cloud from filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if len(pcd.colors) > 0:
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[in_fov_indices])
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=np.array(t))
    # o3d.visualization.draw([filtered_pcd,coordinate_frame],intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix)
    return filtered_pcd

def randomly_add_points_in_point_cloud(N,min_bound,max_bound):
    # min_bound = np.min(np.asarray(pcd.points), axis=0)
    # max_bound = np.max(np.asarray(pcd.points), axis=0)
    # Create N new points randomly distributed in the space defined by min_bound and max_bound
    new_points = np.random.uniform(min_bound, max_bound, size=(N, 3))
    # # Create a new point cloud from the new points
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    # new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    # Visualize the new point cloud
    # o3d.visualization.draw_geometries([new_pcd])

    return new_pcd

def evenly_add_points_in_point_cloud(N,min_bound,max_bound):
    # evenly distributed in the current point cloud space
    x = np.linspace(min_bound[0], max_bound[0], num=N)
    y = np.linspace(min_bound[1], max_bound[1], num=N)
    z = np.linspace(min_bound[2], max_bound[2], num=N)
    # Create a meshgrid, which creates a rectangular grid out of the x, y, and z arrays
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing for Cartesian coordinate ordering
    # Reshape the grids to form a list of coordinates for points in 3D space
    new_pcd = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # new_points
    # new_colors = np.zeros((N, 3))
    return new_pcd

def downsampele_hidden_point_removal(pcd,para_eye,voxel_size=8):
    # downsample and remove hidden points
    centeriod = [0,500,0]
    # get L2 norm of the vector
    radius = np.linalg.norm(np.array(para_eye)-np.array(centeriod))*1000
    # print('radius/1000',radius/1000)
    # downsampling points and remove hidden points
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    _, pt_map = down_pcd.hidden_point_removal(para_eye,radius)
    down_pcd_remove = down_pcd.select_by_index(pt_map)
    return down_pcd_remove

def hidden_point_removal(pcd,para_eye):
    # if pcd has no points, return empty point cloud
    if len(pcd.points) <= 3:
        return pcd
    centroid = [0,500,0]
    # get L2 norm of the vector
    radius = np.linalg.norm(np.array(para_eye)-np.array(centroid))*1000
    # remove hidden points
    _, pt_map = pcd.hidden_point_removal(para_eye,radius)
    pcd_remove = pcd.select_by_index(pt_map)
    return pcd_remove

def downsampele_hidden_point_removal_trace(pcd, para_eye, voxel_size=8, min_bound=None, max_bound=None, approximate_class=False):
    # downsample and remove hidden points
    centeriod = [0,500,0]
    # get L2 norm of the vector
    radius = np.linalg.norm(np.array(para_eye)-np.array(centeriod))*1000
    # downsampling points and remove hidden points
    down_pcd, indices, inverse_indices = pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound, approximate_class=approximate_class)
    # import pdb; pdb.set_trace()
    # get the longest element in the list of inverse_indices
    # print('max:',max(inverse_indices))
    _, pt_map = down_pcd.hidden_point_removal(para_eye, radius)
    down_pcd_remove = down_pcd.select_by_index(pt_map)
    # get the corresponding points in the original point cloud
    # fast merge for a long list
    down_sample_inverse_indices = [inverse_indices[i] for i in pt_map] # we can optimize this by multi-threading
    merged_indices_list = list(chain.from_iterable(down_sample_inverse_indices))
    # concat all list in the inverse_indices
    original_pcd_remove = pcd.select_by_index(merged_indices_list)
    # original_pcd_remove = pcd.select_by_index([inverse_indices[i] for i in pt_map])

    return down_pcd_remove, original_pcd_remove    

def draw_rendering_from_given_FoV_traces(trajectory_positions,trajectory_orientations,
                                trajectory_index,point_cloud_name='longdress'):    
    # get the point cloud data
    data_path = "../point_cloud_data/"
    # shift the pcd to the X,Z plane origin with offset
    if point_cloud_name == 'longdress':
        point_cloud_path = data_path + '8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,147]))#longdress
    elif point_cloud_name == 'loot':
        point_cloud_path = data_path + '8i/loot/loot/Ply/loot_vox10_'+str(1000+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([217,0,231]))#loot
    elif point_cloud_name == 'redandblack':
        point_cloud_path = data_path + '8i/redandblack/redandblack/Ply/redandblack_vox10_'+str(1450+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([365,0,241]))#redandblack
    elif point_cloud_name == 'soldier':
        point_cloud_path = data_path + '8i/soldier/soldier/Ply/soldier_vox10_0'+str(536+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([228,0,198]))#soldier

    
    # get XYZ data
    selected_position = trajectory_positions[trajectory_index]
    para_eye = [i*1024/1.8 for i in selected_position]
    # para_eye = [i*1024/3.4 for i in selected_position]
    para_eye[2] = -para_eye[2]
    para_eye = np.array(para_eye).reshape(3,1)
    # get yaw pitch roll and to orientation
    selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
    pitch_degree, yaw_degree, roll_degree = selected_orientation  # Convert degrees to radians if necessary
    # pitch_degree += 45
    # pitch_degree, yaw_degree, roll_degree = [0,0,0]
    image_width, image_height = np.array([1920, 1080])
    intrinsic_matrix = get_camera_intrinsic_matrix(image_width, image_height)
    # Define camera extrinsic parameters (example values for rotation and translation)
    extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, para_eye)
    # Create a coordinate frame (axis) at the origin, you can adjust the size as needed
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=para_eye)
    
    # o3d.visualization.draw([pcd,coordinate_frame],
    #                        intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
    #                         non_blocking_and_return_uid = False,
    #                        raw_mode=True,
    #                        show_skybox=False,
    #                     #    point_size = 4,
    #                         )
    
    # #  number of points in the point cloud
    original_points = len(pcd.points)
    # print('number of points in the point cloud:',original_points)

    # downsample and remove hidden points
    # pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=1)

    # o3d.visualization.draw([pcd,coordinate_frame],
    #                        intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
    #                         non_blocking_and_return_uid = False,
    #                        raw_mode=True,
    #                        show_skybox=False,
    #                     #    point_size = 4,
    #                         )
    # # number of points in the point cloud
    afterhpr = len(pcd.points)
    # print('number of points in the point cloud after HPR:',afterhpr)

    # pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)

    # o3d.visualization.draw([pcd,coordinate_frame],
    #                        intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
    #                         non_blocking_and_return_uid = False,
    #                        raw_mode=True,
    #                        show_skybox=False,
    #                     #    point_size = 4,
    #                         )    
    # # number of points in the point cloud
    afterfov = len(pcd.points)
    # print('number of points in the point cloud after FoV:',afterfov)

    
    print('draw extrinsic matrix:',extrinsic_matrix,selected_orientation,selected_position,intrinsic_matrix)
    o3d.visualization.draw([pcd,coordinate_frame],
                           intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
                            # non_blocking_and_return_uid = False,
                           raw_mode=True,
                           show_skybox=False,
                        #    point_size = 4,
                            field_of_view=90.0,
                            )
    return original_points,afterhpr,afterfov


# def get_pcd_data(point_cloud_name='longdress', trajectory_index=0):
#     data_path = "../point_cloud_data/"
#     # shift the pcd to the X,Z plane origin with offset
#     if point_cloud_name == 'longdress':
#         point_cloud_path = data_path + '8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
#         pcd = o3d.io.read_point_cloud(point_cloud_path)
#         pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,149]))#longdress
#     elif point_cloud_name == 'loot':
#         point_cloud_path = data_path + '8i/loot/loot/Ply/loot_vox10_'+str(1000+trajectory_index%150)+'.ply'
#         pcd = o3d.io.read_point_cloud(point_cloud_path)
#         pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([217,0,231]))#loot
#     elif point_cloud_name == 'redandblack':
#         point_cloud_path = data_path + '8i/redandblack/redandblack/Ply/redandblack_vox10_'+str(1450+trajectory_index%150)+'.ply'
#         pcd = o3d.io.read_point_cloud(point_cloud_path)
#         pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([365,0,241]))#redandblack
#     elif point_cloud_name == 'soldier':
#         point_cloud_path = data_path + '8i/soldier/soldier/Ply/soldier_vox10_0'+str(536+trajectory_index%150)+'.ply'
#         pcd = o3d.io.read_point_cloud(point_cloud_path)
#         pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([228,0,198]))#soldier
#     return pcd

def get_pcd_data_original(point_cloud_name='longdress', trajectory_index=0):
    data_path = "../point_cloud_data/"
    # shift the pcd to the X,Z plane origin with offset
    if point_cloud_name == 'longdress':
        point_cloud_path = data_path + '8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,149]))#longdress
    elif point_cloud_name == 'loot':
        point_cloud_path = data_path + '8i/loot/loot/Ply/loot_vox10_'+str(1000+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([217,0,231]))#loot
    elif point_cloud_name == 'redandblack':
        point_cloud_path = data_path + '8i/redandblack/redandblack/Ply/redandblack_vox10_'+str(1450+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([365,0,241]))#redandblack
    elif point_cloud_name == 'soldier':
        point_cloud_path = data_path + '8i/soldier/soldier/Ply/soldier_vox10_0'+str(536+trajectory_index%150)+'.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([228,0,198]))#soldier
    return pcd

def get_pcd_data(point_cloud_name='longdress', trajectory_index=0):
    # downsample and remove hidden points
    # 
    if point_cloud_name == 'longdress':
        point_cloud_path = f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    elif point_cloud_name == 'loot':
        point_cloud_path = f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    elif point_cloud_name == 'redandblack':
        point_cloud_path = f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    elif point_cloud_name == 'soldier':
        point_cloud_path = f'./data/{point_cloud_name}/frame{trajectory_index}_downsampled.ply'
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    return pcd
    

def save_rendering_from_given_FoV_traces(trajectory_positions,trajectory_orientations,
                                trajectory_index,point_cloud_name='longdress',user='P03_V1',prefix='',save=False,render_flag=False):    
    pcd = get_pcd_data(point_cloud_name=point_cloud_name, trajectory_index=trajectory_index%150)
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
        pcd = downsampele_hidden_point_removal(pcd,para_eye,voxel_size=8)
        


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
        if not os.path.exists('../result/'+point_cloud_name+'/'+user):
            os.makedirs('../result/'+point_cloud_name+'/'+user)        
        vis.capture_screen_image('../result/'+point_cloud_name+'/'+user+'/'+prefix+'fov_'+str(trajectory_index).zfill(3)+'.png', do_render=False) 
    # index should have 3 digits
    vis.destroy_window()


def get_point_cloud_user_trajectory(pcd_name='longdress', participant='P03_V1'):
    data_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    file_mapping = {
        'longdress': 'H1_nav.csv',
        'loot': 'H2_nav.csv',
        'redandblack': 'H3_nav.csv',
        'soldier': 'H4_nav.csv'
    }
    file_name = file_mapping[pcd_name]
    # if file_name is None:
        # raise ValueError(f"Invalid point cloud name: {pcd_name}")
    positions, orientations = parse_trajectory_data(data_path + file_name, user_index=participant)
    # get the centroid of the positions

    return positions, orientations

def get_point_cloud_user_trajectory_LR(pcd_name='longdress', participant='P03_V1',history=90,future=30):
    data_path = "../point_cloud_data/LR_pred/"
    file_mapping = {
        'longdress': f'H1_nav_pred{history}{future}.csv',
        'loot': f'H2_nav_pred{history}{future}.csv',
        'redandblack': f'H3_nav_pred{history}{future}.csv',
        'soldier': f'H4_nav_pred{history}{future}.csv'
    }
    file_name = file_mapping[pcd_name]
    # if file_name is None:
        # raise ValueError(f"Invalid point cloud name: {pcd_name}")
    positions, orientations = parse_trajectory_data(data_path + file_name, user_index=participant)
    # get the centroid of the positions

    return positions, orientations

def get_point_cloud_user_trajectory_TLR(pcd_name='longdress', participant='P03_V1',history=90,future=30):
    data_path = "../point_cloud_data/TLR_pred/"
    file_mapping = {
        'longdress': f'H1_nav_tlpred{history}{future}.csv',
        'loot': f'H2_nav_tlpred{history}{future}.csv',
        'redandblack': f'H3_nav_tlpred{history}{future}.csv',
        'soldier': f'H4_nav_tlpred{history}{future}.csv'
    }
    file_name = file_mapping[pcd_name]
    # if file_name is None:
        # raise ValueError(f"Invalid point cloud name: {pcd_name}")
    positions, orientations = parse_trajectory_data(data_path + file_name, user_index=participant)
    # get the centroid of the positions

    return positions, orientations

def get_point_cloud_user_trajectory_MLP(pcd_name='longdress', participant='P03_V1',history=90,future=30):
    data_path = "../point_cloud_data/MLP_pred/"
    file_mapping = {
        'longdress': f'H1_nav_MLP_pred{history}{future}.csv',
        'loot': f'H2_nav_MLP_pred{history}{future}.csv',
        'redandblack': f'H3_nav_MLP_pred{history}{future}.csv',
        'soldier': f'H4_nav_MLP_pred{history}{future}.csv'
    }
    file_name = file_mapping[pcd_name]
    # if file_name is None:
        # raise ValueError(f"Invalid point cloud name: {pcd_name}")
    positions, orientations = parse_trajectory_data(data_path + file_name, user_index=participant)
    # get the centroid of the positions

    return positions, orientations

def get_point_cloud_user_trajectory_LSTM(pcd_name='longdress', participant='P03_V1',history=90,future=30):
    data_path = "../point_cloud_data/LSTM_pred/"
    file_mapping = {
        'longdress': f'H1_nav_LSTM_pred{history}{future}.csv',
        'loot': f'H2_nav_LSTM_pred{history}{future}.csv',
        'redandblack': f'H3_nav_LSTM_pred{history}{future}.csv',
        'soldier': f'H4_nav_LSTM_pred{history}{future}.csv'
    }
    file_name = file_mapping[pcd_name]
    # if file_name is None:
        # raise ValueError(f"Invalid point cloud name: {pcd_name}")
    positions, orientations = parse_trajectory_data(data_path + file_name, user_index=participant)
    # get the centroid of the positions

    return positions, orientations

if __name__ == '__main__':
#   the following code is for testing the in-FoV and draw intrinsic and extrinsic matrix functions

    # Load your point cloud
    point_cloud_path = '../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1051.ply'
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # Define camera intrinsic parameters (example values)
    image_width, image_height = np.array([1280, 720])
    intrinsic_matrix = get_camera_intrinsic_matrix(image_width, image_height)
    # Define camera extrinsic parameters (example values for rotation and translation)
    yaw_degree, pitch_degree, roll_degree = 0, 0, 0 # this is left-hand coordinate system
    t = np.array([[200], [800], [500]]) # Translation
    extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll(yaw_degree, pitch_degree, roll_degree, t)
    # get point in FoV
    # filtered_pcd = get_points_in_FoV(pcd, intrinsic_matrix, extrinsic_matrix, image_width, image_height)

    # downsample and remove hidden points
    # get the minimum and maximum bound of the point cloud
    min_bound = np.min(np.asarray(pcd.points), axis=0)
    max_bound = np.max(np.asarray(pcd.points), axis=0)

    down_pcd_remove, original_pcd_remove = downsampele_hidden_point_removal_trace(pcd, t, 
                    voxel_size=1, min_bound=min_bound, max_bound=max_bound, approximate_class=False)
    # Visualize the filtered point cloud
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
    o3d.visualization.draw([original_pcd_remove,coordinate_frame],
                           intrinsic_matrix=intrinsic_matrix,extrinsic_matrix=extrinsic_matrix,
                           raw_mode=True,show_skybox=False)
    
    # # randomly add 100 new points in the current point cloud uniformly distributed in the current point cloud space
    # new_pcd = randomly_add_points_in_point_cloud(100000,min_bound,max_bound)
    # pcd += new_pcd
    # # save the filtered point cloud to ply file
    # # o3d.io.write_point_cloud("./result/FoV_filtered_point_cloud_example.ply", filtered_pcd, write_ascii=True)

#  the following code is for testing the rendering and saving function
    # positions,orientations = get_point_cloud_user_trajectory(pcd_name='longdress',participant='P03_V1')
    # draw_rendering_from_given_FoV_traces 
    # afterfov_list = []
    # afterhpr_list = []
    # original_list = []
    # end_index = len(positions)
    # end_index = 1
    # for index in range(0, end_index,1):
    #     print('index:',index)
    #     original,afterhpr,afterfov  = draw_rendering_from_given_FoV_traces(positions,orientations,
    #                             trajectory_index=index,point_cloud_name=pcd_name) 
    #     # original_list.append(original)
    #     afterhpr_list.append(afterhpr)
    #     afterfov_list.append(afterfov)
    # print('average original_list:',sum(original_list)/len(original_list))
    # print('average afterhpr_list:',sum(afterhpr_list)/len(afterhpr_list))
    # print('average afterfov_list:',sum(afterfov_list)/len(afterfov_list))

        
    # save the renderred image to a file
    # check open3d version
    # print(o3d.__version__)
#     for pcd_name in ['loot','redandblack','soldier']:
# # for pcd_name in ['longdress','loot','redandblack','soldier']:        
#         for user in ['P08_V1','P03_V1','P01_V1']:
#         # for user in ['P12_V1']:
#             positions,orientations = get_point_cloud_user_trajectory(pcd_name=pcd_name,participant=user)
#             end_index = len(positions)
#             # end_index = 1
#             for index in range(0, end_index,1):
#                 print('index:',index)
#                 save_rendering_from_given_FoV_traces(positions,orientations,
#                                         trajectory_index=index,point_cloud_name=pcd_name,user=user,prefix='all_points')         


    # positions,orientations = get_point_cloud_user_trajectory(pcd_name='longdress',participant='P03_V1')
    # end_index = len(positions)
    # print('end_index:',end_index)
    # # end_index = 1
    # for index in range(0, end_index,1):
    #     print('index:',index)
    #     save_rendering_from_given_FoV_traces(positions,orientations,
    #                             trajectory_index=index,point_cloud_name=pcd_name,user=participant)
        
    
