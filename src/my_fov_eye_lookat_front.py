from node_feature_utils import parse_trajectory_data
import numpy as np
from node_feature_utils import *
trajectory_positions, trajectory_orientations = parse_trajectory_data("./6DoF-HMD-UserNavigationData-master/NavigationData/H1_nav.csv",user_index='P01_V1')
def rotate_vector(vector, axis, angle):
    """
    Rotate a vector around a given axis by an angle in radians.
    Uses Rodrigues' rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    vector_rotated = (vector * np.cos(angle) +
                      np.cross(axis, vector) * np.sin(angle) +
                      axis * np.dot(axis, vector) * (1 - np.cos(angle)))
    return vector_rotated

def get_up_vector_from_roll(roll_angle_radians, front_vector):
    """
    Calculate the "up" vector given the roll angle and the front vector of the camera.
    Assumes roll_angle_degrees is given in degrees and will be converted to radians.
    """
    # Convert roll angle to radians
    # roll_angle_radians = np.radians(roll_angle_degrees)
    
    # Initial "up" vector before any rotation
    initial_up = np.array([0, 1, 0])
    
    # Apply roll rotation around the front vector
    up_vector = rotate_vector(initial_up, front_vector, roll_angle_radians)
    
    return up_vector

# %run node_feature_utils.py
def rendering_from_node_feature(trajectory_positions,trajectory_orientations,
                                trajectory_index):    
    
    point_cloud_path = '8i/longdress/longdress/Ply/longdress_vox10_'+str(1051+trajectory_index%150)+'.ply'
    # point_cloud_path = '8i/loot/loot/Ply/loot_vox10_'+str(1000+trajectory_index%150)+'.ply'
    # point_cloud_path = '8i/redandblack/redandblack/Ply/redandblack_vox10_'+str(1450+trajectory_index%150)+'.ply'
    # point_cloud_path = '8i/soldier/soldier/Ply/soldier_vox10_0'+str(536+trajectory_index%150)+'.ply'
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # change pcd's z axis to -z
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) * np.array([1,1,-1]))
    # shift the pcd to the X,Z plane origin with offset
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([246,0,147]))#longdress
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([217,0,231]))#loot
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([365,0,241]))#redandblack   
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - np.array([228,0,198]))#soldier


    # get XYZ data
    selected_position = trajectory_positions[trajectory_index]
    para_eye = [i*1024/1.8 for i in selected_position]
    para_eye[2] = -para_eye[2]
    print(para_eye)

    # pcd_list = []

    # get yaw pitch roll and to orientation
    selected_orientation = trajectory_orientations[trajectory_index]  # First orientation (yaw, pitch, roll)
    pitch, yaw, roll = np.radians(selected_orientation)  # Convert degrees to radians if necessary
    # Calculate the direction vector from the orientation
    orientation = euler_to_direction(yaw, pitch, roll)
    orientation[0] = 0
    
    para_lookat = [para_eye[i]+orientation[i] for i in range(3)]
    
    roll_angle_radians = roll
    front_vector = orientation
    # normalize front_vector to be the unit vector
    front_vector = front_vector / np.linalg.norm(front_vector)
    my_up = get_up_vector_from_roll(roll_angle_radians, front_vector)
    # print('my_up',my_up)
    
    print('eye',para_eye)
    print('lookat',para_lookat)
    print('orientation',orientation)
    print('roll',roll)
    print('up',my_up)# it is not orthogonal to orientation, weird
    # write a code to verify the my_up vector is orthogonal to front_vector
    np.dot(front_vector,my_up)
# downsample and remove hidden points
    centeriod = [300,500,200]
    # radius is Euclidean distance from camera to the centeriod
    # radius = 1000*np.sqrt((camera[0]-centeriod[0])**2 + (camera[1]-centeriod[1])**2 + (camera[2]-centeriod[2])**2)
    # get L2 norm of the vector
    radius = np.linalg.norm(np.array(para_eye)-np.array(centeriod))*1000
    print('radius/1000',radius/1000)

    # downsampling points and remove hidden points
    # voxel_size = 1
    # down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # _, pt_map = down_pcd.hidden_point_removal(para_eye,radius)
    # down_pcd_remove = down_pcd.select_by_index(pt_map)
    # pcd = down_pcd_remove
    # 

    # Create a coordinate frame (axis) at the origin, you can adjust the size as needed
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0, 0, 0])


    o3d.visualization.draw([pcd,coordinate_frame],lookat=para_lookat,eye=para_eye,up=my_up)
    
    # export the image to a file
    


    # print(para_eye,para_lookat)

for index in range(2,10):
    rendering_from_node_feature(trajectory_positions,trajectory_orientations,
                                trajectory_index=index)
    
# rendering_from_node_feature(trajectory_positions,trajectory_orientations,
                                # trajectory_index=10)