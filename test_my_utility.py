from point_cloud_FoV_utils import *
import pandas as pd
import matplotlib.pyplot as plt

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

def read_node_feature():
    node_feature_path = './data/longdress_P01_V1_VS128/node_feature.csv'
    node_feature = pd.read_csv(node_feature_path)
    oo = node_feature['occlusion_feature']
    oo = node_feature['in_FoV_feature']
    print(oo)
    # get histogram of occlusion feature and print the statistics
    plt.hist(oo, bins=100)
    plt.show()
    print('Mean:',np.mean(oo))
    print('Std:',np.std(oo))
    print('Max:',np.max(oo))
    print('Min:',np.min(oo))
    print('Done')

def draw_loss():
    loss ='''   epoch:0,  loss: 0.04622
epoch:1,  loss: 0.03837
epoch:2,  loss: 0.03578
epoch:3,  loss: 0.03398
epoch:4,  loss: 0.03161
epoch:5,  loss: 0.02971
epoch:6,  loss: 0.02799
epoch:7,  loss: 0.02633
epoch:8,  loss: 0.02467
epoch:9,  loss: 0.02287
epoch:10,  loss: 0.02089
epoch:11,  loss: 0.01886
epoch:12,  loss: 0.01695
epoch:13,  loss: 0.01538
epoch:14,  loss: 0.01415
epoch:15,  loss: 0.01321
epoch:16,  loss: 0.01244
epoch:17,  loss: 0.01171
epoch:18,  loss: 0.01103
epoch:19,  loss: 0.01040
epoch:20,  loss: 0.00985
epoch:21,  loss: 0.00937
epoch:22,  loss: 0.00891
epoch:23,  loss: 0.00849
epoch:24,  loss: 0.00813
epoch:25,  loss: 0.00781
epoch:26,  loss: 0.00752
epoch:27,  loss: 0.00727
epoch:28,  loss: 0.00706
epoch:29,  loss: 0.00687
epoch:30,  loss: 0.00671
epoch:31,  loss: 0.00657
epoch:32,  loss: 0.00646
epoch:33,  loss: 0.00636
epoch:34,  loss: 0.00627
epoch:35,  loss: 0.00620
epoch:36,  loss: 0.00615
epoch:37,  loss: 0.00610
epoch:38,  loss: 0.00605
epoch:39,  loss: 0.00601
epoch:40,  loss: 0.00597
epoch:41,  loss: 0.00593
epoch:42,  loss: 0.00590
epoch:43,  loss: 0.00587
epoch:44,  loss: 0.00583
epoch:45,  loss: 0.00579
epoch:46,  loss: 0.00575
epoch:47,  loss: 0.00571
epoch:48,  loss: 0.00568
epoch:49,  loss: 0.00565
epoch:50,  loss: 0.00564
epoch:51,  loss: 0.00563
epoch:52,  loss: 0.00562
epoch:53,  loss: 0.00561
epoch:54,  loss: 0.00560
epoch:55,  loss: 0.00560
epoch:56,  loss: 0.00559
epoch:57,  loss: 0.00559
epoch:58,  loss: 0.00558
epoch:59,  loss: 0.00558
epoch:60,  loss: 0.00556
epoch:61,  loss: 0.00554
epoch:62,  loss: 0.00551
epoch:63,  loss: 0.00550
epoch:64,  loss: 0.00550
epoch:65,  loss: 0.00548
epoch:66,  loss: 0.00546
epoch:67,  loss: 0.00544
epoch:68,  loss: 0.00543
epoch:69,  loss: 0.00542
epoch:70,  loss: 0.00541
epoch:71,  loss: 0.00541
epoch:72,  loss: 0.00541
epoch:73,  loss: 0.00540
epoch:74,  loss: 0.00539
epoch:75,  loss: 0.00539
epoch:76,  loss: 0.00538
epoch:77,  loss: 0.00537
epoch:78,  loss: 0.00536
epoch:79,  loss: 0.00535
epoch:80,  loss: 0.00535
epoch:81,  loss: 0.00534
epoch:82,  loss: 0.00534
epoch:83,  loss: 0.00533
epoch:84,  loss: 0.00533
epoch:85,  loss: 0.00532
epoch:86,  loss: 0.00532
epoch:87,  loss: 0.00532
epoch:88,  loss: 0.00531
epoch:89,  loss: 0.00530
epoch:90,  loss: 0.00530
epoch:91,  loss: 0.00529
epoch:92,  loss: 0.00529
epoch:93,  loss: 0.00529
epoch:94,  loss: 0.00528
epoch:95,  loss: 0.00528
epoch:96,  loss: 0.00527
epoch:97,  loss: 0.00527
epoch:98,  loss: 0.00526
epoch:99,  loss: 0.00526'''
    loss = loss.split('\n')
    loss = [i.split(',') for i in loss]
    # import pdb; pdb.set_trace()
    loss = [i[1].split(':') for i in loss]
    loss = [float(i[1]) for i in loss]
    plt.plot(loss)
    plt.show()



if __name__ == '__main__':
    # test_voxel_size()
    # test_FoV_centroid()
    # test_pcd_centroid()
    # test_pcd_min_max_bound()
    # for depth in range(3, 10 ,3):
        # octree_test(depth)
    # octree_test(6)
    # read_node_feature()
    draw_loss()
    print('Done')