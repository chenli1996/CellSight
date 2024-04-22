from point_cloud_FoV_utils import *


def test_voxel_size():
    pcd_name ='longdress'
    user = 'P01_V1'

    for index in range(0, 300,30):
        print('index:',index)
        positions,orientations = get_point_cloud_user(pcd_name=pcd_name,participant=user)
        save_rendering_from_given_FoV_traces(positions,orientations,trajectory_index=index,point_cloud_name=pcd_name,user=user,render_flag=True)

def test_FoV_centroid():
    pcd_name ='longdress'
    # loop for all pcd names

    # loop for all pcd names
    pcd_names =  ['longdress','loot','redandblack','soldier']  # Add all the pcd names here

    for pcd_name in pcd_names:
        # get the final centroid for all 26 participants's position and orientation
        # positions_all = []
        for participant in range(1, 2):
            user = f'P{participant:02d}_V1'
            positions, orientations = get_point_cloud_user(pcd_name=pcd_name, participant=user)
            # centroid = np.mean(positions, axis=0)
            # concatenate all the positions
            if participant == 1:
                positions_all = positions
            else:
                positions_all = np.concatenate((positions_all, positions), axis=0)
            # print(f'Centroid for participant {participant}: {centroid}')
        positions_all = np.array(positions_all)
        final_centroid = np.mean(positions_all, axis=0)
        print(f'Final centroid for {pcd_name}: {final_centroid}')

def test_pcd_centroid():
    pcd_name ='longdress'
    # loop for all pcd names
    for trajectory_index in range(0, 150):
        pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
        centroid = np.mean(np.array(pcd.points), axis=0)
        print(f'Centroid for {pcd_name}: {centroid}')

def test_pcd_min_max_bound():
    # each video has 150 frames and get the min and max bound for the whole video
    # pcd_name ='longdress'
    for pcd_name in ['longdress','loot','redandblack','soldier']:
        min_bounds = []
        max_bounds = []
        for trajectory_index in range(0, 150):
            pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=trajectory_index)
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)
            # print(f'Min bound for {pcd_name} index {trajectory_index}: {min_bound}')
            # print(f'Max bound for {pcd_name} index {trajectory_index}: {max_bound}')
        min_bounds = np.array(min_bounds)
        max_bounds = np.array(max_bounds)
        min_bound = np.min(min_bounds, axis=0)
        max_bound = np.max(max_bounds, axis=0)
        print(f'Min bound for {pcd_name}: {min_bound}')
        print(f'Max bound for {pcd_name}: {max_bound}')
        # Min bound for longdress: [-224.    0. -147.]
        # Max bound for longdress: [ 235. 1023.  513.]
        # Min bound for loot: [-217.    0. -231.]
        # Max bound for loot: [ 197. 1023.  243.]
        # Min bound for redandblack: [-251.    0. -241.]
        # Max bound for redandblack: [ 251. 1008.  251.]
        # Min bound for soldier: [-204.    0. -168.]
        # Max bound for soldier: [ 262. 1023.  311.]
        # overall bound: z -= 1
        # Min bound for all: [-251.    0. -242.]
        # Max bound for all: [ 262. 1023.  512.]
        
        



if __name__ == '__main__':
    # test_voxel_size()
    # test_FoV_centroid()
    # test_pcd_centroid()
    test_pcd_min_max_bound()
    print('Done')