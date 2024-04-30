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
        # Min bound for all: [-251.    0. -241.]
        # Max bound for all: [ 262. 1023.  511.]
        
        
def octree_test(depth=9):
    pcd_name ='longdress'
    pcd = get_pcd_data(point_cloud_name=pcd_name, trajectory_index=0)
    octree = o3d.geometry.Octree(max_depth=depth)
    octree.convert_from_point_cloud(pcd)
    # # get the different levels of the octree and visualize as a point cloud
    # voxel_grid_points = []
    # for i in range(depth):
    #     voxel_grid_points += octree.get_voxel_centers(i)
    # voxel_grid_points = np.array(voxel_grid_points)
    # print(f'Number of voxel grid points: {voxel_grid_points.shape}')
    


    # visualize the voxel grid points as a point cloud
    # voxel_pcd = o3d.geometry.PointCloud()
    # voxel_pcd.points = o3d.utility.Vector3dVector(voxel_grid_points)
    # visualize the voxel grid points
    # o3d.visualization.draw_geometries([voxel_pcd])
    # visualize the octree
    o3d.visualization.draw_geometries([octree])
    # o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':
    # test_voxel_size()
    # test_FoV_centroid()
    # test_pcd_centroid()
    # test_pcd_min_max_bound()
    for depth in range(3, 10 ,3):
        octree_test(depth)
    # octree_test(6)
    print('Done')