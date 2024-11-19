# from point_cloud_FoV_Graph import baseline_loss
from voxel_grid import *
from point_cloud_FoV_utils import *
import pandas as pd
from tqdm import tqdm
from FSVVD_data_utils import *


# write a function to get graph edges, which is node index pair, based on the voxel grid index set, the graph is a 3D grid graph
def get_graph_edges(original_index_to_integer_index,graph_voxel_grid_coords):
    graph_edges = []
    graph_edges_ingeter = []
    for index in original_index_to_integer_index:
        x,y,z = index
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i==0 and j==0 and k==0:
                        continue
                    if abs(i)+abs(j)+abs(k) == 0:
                        continue
                    if abs(i)+abs(j)+abs(k) == 1:
                        edge_feature = 1 #side adjecent
                    if abs(i)+abs(j)+abs(k) == 2:
                        edge_feature = 2 #edge adjecent
                    if abs(i)+abs(j)+abs(k) == 3:
                        edge_feature = 3 #corner adjecent
                    if (x+i,y+j,z+k) in original_index_to_integer_index:
                        graph_edges.append((index,(x+i,y+j,z+k),edge_feature))            
                        graph_edges_ingeter.append((original_index_to_integer_index[index],
                                                    original_index_to_integer_index[(x+i,y+j,z+k)],
                                                    edge_feature))                     
    # reshape the graph_edges to the (n,2) shape
    # graph_edges = np.array(graph_edges).reshape(-1,2*3+1)
    graph_edges_ingeter = np.array(graph_edges_ingeter).reshape(-1,3)
    return graph_edges,graph_edges_ingeter

def generate_graph(voxel_size=0.5):   
    image_width, image_height = np.array([1920, 1080])
    # generate graph voxel grid features

    # voxel_size = int(64)
    min_bounds = np.array([-2.82,  -0.78  , -2.31]) 
    max_bounds = np.array([ 1.32, 2.08,  1.94])

    # min_bounds = np.array([-1.05,  -0.81  , -1.22])
    # max_bounds = np.array([ 1.0, 0.84,  1.25])
    
    edge_prefix = str(voxel_size) + 'fsvvd_raw'
    # get the graph max and min bounds
    # graph_max_bound,graph_min_bound,graph_voxel_grid_integer_index_set,graph_voxel_grid_index_set,graph_voxel_grid_coords,original_index_to_integer_index = voxelizetion_para(
        # voxel_size=voxel_size, min_bounds=min_bounds, max_bounds=max_bounds)
    results = voxelizetion_para(voxel_size=voxel_size, min_bounds=min_bounds, 
                                max_bounds=max_bounds)
    graph_max_bound = results['graph_voxel_grid_max_bound']
    graph_min_bound = results['graph_voxel_grid_min_bound']
    graph_voxel_grid_integer_index_set = results['graph_voxel_grid_integer_index_set']
    graph_voxel_grid_index_set = results['graph_voxel_grid_index_set']
    graph_voxel_grid_coords = results['graph_voxel_grid_coords']
    graph_voxel_grid_coords_array = results['graph_voxel_grid_coords_array']
    original_index_to_integer_index = results['original_index_to_integer_index']
    # if graph_edges_integer_index.csv exists, then load the graph_edges from the csv file
    if os.path.exists(f'./data/{edge_prefix}/graph_edges_integer_index_xxxxxx.csv'):
        graph_edges_df_integer = pd.read_csv(f'./data/{edge_prefix}/graph_edges_integer_index.csv')
        # graph_edges_ingeter = graph_edges_df_integer.values
        graph_edges_ingeter = graph_edges_df_integer[['start_node','end_node','edge_feature']].values
        graph_edges_df = pd.read_csv(f'./data/{edge_prefix}/graph_edges_voxel_index.csv')
        graph_edges = graph_edges_df[['start_node','end_node','edge_feature']].values
    else:
        # mkdir the data folder
        if not os.path.exists(f'./data/{edge_prefix}'):
            os.makedirs(f'./data/{edge_prefix}')
        graph_edges,graph_edges_ingeter = get_graph_edges(original_index_to_integer_index,graph_voxel_grid_coords)
        graph_edges_df = pd.DataFrame(graph_edges,columns=['start_node','end_node','edge_feature'])
        graph_edges_df.to_csv(f'./data/{edge_prefix}/graph_edges_voxel_index.csv',index=False) 
        # save the graph_edge_integer to the csv file with the format of (start_node,end_node,edge_feature) and column names is 'start_node','end_node','edge_feature'
        graph_edges_df_integer = pd.DataFrame(graph_edges_ingeter,columns=['start_node','end_node','edge_feature'])
        graph_edges_df_integer.to_csv(f'./data/{edge_prefix}/graph_edges_integer_index.csv')

    # print(graph_edges_df)
    # print(graph_edges_df_integer)

def generate_node_feature(baseline='GT'):
    # participant = 'P01_V1'
    # trajectory_index = 0
    image_width, image_height = np.array([1920, 1080])
    # generate graph voxel grid features
    voxel_size = 0.6
    min_bounds = np.array([-2.82,  -0.78  , -2.31]) 
    max_bounds = np.array([ 1.32, 2.08,  1.94])

    # voxel_size = 0.4
    # min_bounds = np.array([-1.05,  -0.81  , -1.22])
    # max_bounds = np.array([ 1.0, 0.84,  1.25])
    
    # edge_prefix = str(voxel_size)
    # get the graph max and min bounds
    # graph_max_bound,graph_min_bound,graph_voxel_grid_integer_index_set,graph_voxel_grid_index_set,graph_voxel_grid_coords,original_index_to_integer_index = voxelizetion_para(
        # voxel_size=voxel_size, min_bounds=min_bounds, max_bounds=max_bounds)
    results = voxelizetion_para(voxel_size=voxel_size, min_bounds=min_bounds, 
                                max_bounds=max_bounds)
    graph_max_bound = results['graph_voxel_grid_max_bound']
    graph_min_bound = results['graph_voxel_grid_min_bound']
    graph_voxel_grid_integer_index_set = results['graph_voxel_grid_integer_index_set']
    graph_voxel_grid_index_set = results['graph_voxel_grid_index_set']
    graph_voxel_grid_coords = results['graph_voxel_grid_coords']
    graph_voxel_grid_coords_array = results['graph_voxel_grid_coords_array']
    original_index_to_integer_index = results['original_index_to_integer_index']
    # baseline = 'GT'
    # baseline = 'LSTM'
    history = 90
    if baseline == 'LR':
        history = 30
    print(f'Processing {baseline}...')
    if baseline == 'GT':
        user_list = ['ChenYongting','GuoYushan','Guozhaonian','HKY','RenZhichen','Sunqiran','WangYan','fupingyu','huangrenyi','liuxuya','sulehan','yuchen']
        pcd_name_list = ['Chatting','Pulling_trolley','News_interviewing','Sweep']
        future_list = [0]
    else:# testing set only
        user_list = ['ChenYongting', 'GuoYushan', 'Guozhaonian', 'HKY', 'RenZhichen', 'Sunqiran']
        pcd_name_list = ['Sweep']
        future_list = [150,60,30,10,1]
        # future_list = [1]

    for pcd_name in pcd_name_list:
        
        if baseline == 'GT':
            prefix = f'{pcd_name}_VS{voxel_size}'
        else:
            prefix = f'{pcd_name}_VS{voxel_size}_{baseline}'
        # prefix = f'{pcd_name}_VS{voxel_size}_LR' # LR is _LR for testing***********************************************
        # prefix = f'{pcd_name}_VS{voxel_size}_TLR' # LR is _LR for testing***********************************************
        # prefix = f'{pcd_name}_VS{voxel_size}_MLP' # MLP is _MLP for testing***********************************************
        # prefix = f'{pcd_name}_VS{voxel_size}_LSTM' # LSTM is _LSTM for testing***********************************************
        
        # prefix = f'{pcd_name}_VS{voxel_size}_filtered'
        # for future in [60]:
        for future in future_list:
        # for future in [150]: # for ground truth, it does not matter the future value, only use one future value
            print(f'Processing {pcd_name} with history {history} and future {future}...')
            # for user_i in tqdm(range(1,15)):  # TLP/LR/MLP/LSTM is 15 for testing***********************************************
            # for user_i in tqdm(range(1,28)):    
            for user_i in tqdm(user_list):            
            # for user_i in tqdm(range(1,2)):                
                # participant = 'P'+str(user_i).zfill(2)+'_V1'
                node_index = []
                occupancy_feature = []
                in_FoV_feature = []
                occlusion_feature = []
                distance_feature = []
                coordinate_feature = []
                # choose different trajectory files***********************************************
                if baseline == 'GT':
                    positions,orientations = get_point_cloud_user_trajectory_FSVVD(pcd_name=pcd_name,participant=user_i)
                else:
                    positions,orientations = get_point_cloud_user_trajectory_FSVVD_baseline(baseline=baseline,pcd_name=pcd_name,participant=user_i,history=history,future=future)
                # import pdb; pdb.set_trace()
                # positions,orientations = get_point_cloud_user_trajectory_LR(pcd_name=pcd_name,participant=participant,history=history,future=future) # LR is _LR for testing***********************************************
                # positions,orientations = get_point_cloud_user_trajectory_TLR(pcd_name=pcd_name,participant=participant,history=history,future=future) # TLR is _TLR for testing***********************************************
                # positions,orientations = get_point_cloud_user_trajectory_MLP(pcd_name=pcd_name,participant=participant,history=history,future=future) # MLP is _MLP for testing***********************************************
                # positions,orientations = get_point_cloud_user_trajectory_LSTM(pcd_name=pcd_name,participant=participant,history=history,future=future) # MLP is _MLP for testing***********************************************
                sample_step = 2
                for trajectory_index in tqdm(range(0,(len(positions)),sample_step),desc=f'Processing {user_i}'):
                    # print(f'Processing trajectory {trajectory_index}...')
                    # Load the point cloud data
                    # pcd = get_pcd_data_FSVVD(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
                    # pcd = get_pcd_data_FSVVD_downsampled(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
                    pcd = get_pcd_data_FSVVD_downsampled_fsvvd(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
                    # pcd = get_pcd_data_FSVVD_filtered(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
                    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
                    # get the position and orientation for the given participant and trajectory index
                    # import pdb; pdb.set_trace()
                    position = positions[trajectory_index]
                    orientation = orientations[trajectory_index]
                    para_eye = position
                    # para_eye = np.array(para_eye).reshape(3,1)
                    pitch_degree, yaw_degree, roll_degree = orientation
                    
                    # Define camera intrinsic parameters
                    intrinsic_matrix = get_camera_intrinsic_matrix_fsvvd(image_width, image_height)
                    # Define camera extrinsic parameters
                    extrinsic_matrix = get_camera_extrinsic_matrix_from_yaw_pitch_roll_fsvvd(yaw_degree, pitch_degree, roll_degree, para_eye)



                    # pcd = pcd.voxel_down_sample(voxel_size=8)
                    # get the occupancy feature
                    occupancy_dict,occupancy_array = get_occupancy_feature(pcd,graph_min_bound,graph_max_bound,graph_voxel_grid_integer_index_set,voxel_size)
                    # print('occupancy_dict:      ',occupancy_dict[(1, 5, 2)])
                    

                    # get the in_FoV_voxel_percentage_dict
                    in_FoV_percentage_dict,in_FoV_voxel_percentage_array,pcd_N = get_in_FoV_feature(graph_min_bound,graph_max_bound,voxel_size,intrinsic_matrix,extrinsic_matrix,image_width,image_height)
                    # print('in_FoV_dict:         ',in_FoV_percentage_dict[(1, 5, 2)])

                    # get occlusion level
                    # deep copy the pcd
                    
                    occlusion_level_dict,occulusion_array,pcd_hpr = get_occlusion_level_dict(pcd,para_eye,graph_min_bound,graph_max_bound,graph_voxel_grid_integer_index_set,voxel_size,intrinsic_matrix,extrinsic_matrix,image_width,image_height)
                    # print('occlusion_level_dict:',occlusion_level_dict[(2, 0, 2)])
                    # print('occupancy_dict:      ',occupancy_dict)
                    # print('occupancy_array:      ',occupancy_array)
                    # print('occlusion_level_dict:',occlusion_level_dict)
                    # print('occulusion_array:    ',occulusion_array)
                    # print('in_FoV_dict:         ',in_FoV_percentage_dict)
                    # print('in_FoV_array:        ',in_FoV_voxel_percentage_array)
                    # visualize the voxel grid
                    # visualize_voxel_grid(pcd,pcd_hpr,graph_min_bound,graph_max_bound,voxel_size,para_eye,graph_voxel_grid_integer_index_set,graph_voxel_grid_coords)
                    # append features
                    occupancy_feature.append(occupancy_array)
                    in_FoV_feature.append(in_FoV_voxel_percentage_array)
                    occlusion_feature.append(occulusion_array)
                    node_index.append(graph_voxel_grid_integer_index_set)
                    coordinate_feature.append(graph_voxel_grid_coords_array)
                    distance_feature.append(np.linalg.norm(graph_voxel_grid_coords_array-para_eye,axis=1).reshape(-1,1))
                # save the features to the csv file
                occupancy_feature = np.array(occupancy_feature).reshape(-1,1)
                in_FoV_feature = np.array(in_FoV_feature).reshape(-1,1)
                occlusion_feature = np.array(occlusion_feature).reshape(-1,1)
                node_index = np.array(node_index).reshape(-1,1)
                coordinate_feature = np.array(coordinate_feature).reshape(-1,3)
                distance_feature = np.array(distance_feature).reshape(-1,1)
                theta_feature = 2 * np.arctan(occlusion_feature * voxel_size / (2 * distance_feature))
                node_feature = np.concatenate((occupancy_feature,in_FoV_feature,occlusion_feature,theta_feature,coordinate_feature,distance_feature,node_index),axis=1)
                node_feature_df = pd.DataFrame(node_feature,columns=['occupancy_feature','in_FoV_feature','occlusion_feature','theta_feature','coordinate_x','coordinate_y','coordinate_z','distance','node_index'])                                
                if not os.path.exists(f'./data/{prefix}'):
                    os.makedirs(f'./data/{prefix}')
                if baseline == 'GT':
                    node_feature_df.to_csv(f'./data/{prefix}/{user_i}node_feature.csv')
                    print(f'saved to file /data/{prefix}/{user_i}node_feature.csv')
                else:
                 # LR for testing***********************************************
                    node_feature_df.to_csv(f'./data/{prefix}/{user_i}node_feature{history}{future}.csv')
                # save to 
                    print(f'saved to file /data/{prefix}/{user_i}node_feature{history}{future}.csv')




if __name__ == '__main__':
    # generate_graph(voxel_size=0.6) # run once to generate graph edges
    for baseline in ['LR']:
    # for baseline in ['GT','TLP','LR','MLP','LSTM']:
        generate_node_feature(baseline=baseline)
    # generate_node_feature()
    # downsample_binary_pcd_data()

# test pull rebase

    