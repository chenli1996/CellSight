import open3d as o3d
import json
import time
import numpy as np
from itertools import chain
# Assuming `pcd` is your point cloud object.
# Load or create your point cloud here
# pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")
def main():
    point_cloud_path = '../point_cloud_data/8i/longdress/longdress/Ply/longdress_vox10_1051.ply'
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # get the subset of the point cloud
    # pcd = pcd.select_by_index(np.arange(0, 100000))

    camera = [ 2000, 1000, 200 ]
    centeriod = [300,500,200]

    # radius is Euclidean distance from camera to the centeriod
    # radius = 1000*np.sqrt((camera[0]-centeriod[0])**2 + (camera[1]-centeriod[1])**2 + (camera[2]-centeriod[2])**2)
    # get L2 norm of the vector
    radius = np.linalg.norm(np.array(camera)-np.array(centeriod))*1000
    print('radius/1000',radius/1000)

    # downsampling points to 10 times less
    voxel_size = 8
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print('voxel_size:',voxel_size)
    # voxel_size = 1 - 32s
    # voxel_size = 4 - 2.4s
    # voxel_size = 8 - 0.8s
    # voxel_size = 16 - 0.4s


    # get the running time of the hidden_point_removal    
    start_time = time.time()
    down_pcd_test_time = pcd.voxel_down_sample(voxel_size=voxel_size)
    for i in range(30):
        # down_pcd_test_time = pcd.voxel_down_sample(voxel_size=voxel_size)
        _, pt_map = down_pcd_test_time.hidden_point_removal(camera,radius)
    print("Time taken for hidden_point_removal on 30 frames: ", time.time()-start_time)

    # get the running time of the downsampling
    start_time = time.time()
    for i in range(30):
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # _, pt_map = down_pcd.hidden_point_removal(camera,radius)
    print("Time taken for downsampling on 30 frames: ", time.time()-start_time)


    # get the running time of getting the point from hpr index
    start_time = time.time()
    for i in range(30):
        down_pcd_remove = down_pcd.select_by_index(pt_map)
    print("Time taken for getting the point from hpr index on 30 frames: ", time.time()-start_time)


    # get the running time of the hidden_point_removal with downsample trace

    # get min_bound and max_bound
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    start_time = time.time()
    for i in range(30):
        down_pcd_test_time, indices, inverse_indices = pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound)
        _, pt_map = down_pcd_test_time.hidden_point_removal(camera,radius)
        # down_sample_inverse_indices = [inverse_indices[i] for i in pt_map] # we can optimize this by multi-threading
        # merged_indices_list = list(chain.from_iterable(down_sample_inverse_indices))
        # original_pcd_remove = pcd.select_by_index(merged_indices_list)
    print("Time taken for hidden_point_removal with trace on 30 frames with downsample: ", time.time()-start_time)

    return
def test_hpr_smallest_number_of_points():
    # build a point cloud with 2 points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0],[1,1,1],[1,10,2]]))
    camera = [ 0, 0, 100 ]
    radius = 1000000*2
    _, pt_map = pcd.hidden_point_removal(camera,radius)
    print('pt_map:',pt_map)

if __name__ == '__main__':
    # main()
    test_hpr_smallest_number_of_points()

    # radius = 1000000*2
    # pcd = down_pcd
    # # print("Get all points that are visible from given view point")
    # # front = [ 0.97077843081593018, 0.077085970480618979, 0.22725974438430921 ]
    # # lookat = [ 281.0, 509.5, 174.5 ]
    # # up = [ -0.074292602323873899, 0.99701885334845908, -0.020833034048614155 ]
    # _, pt_map = pcd.hidden_point_removal(camera,radius)

    # # print("Visualize result")
    # pcd_new = pcd.select_by_index(pt_map)
    # # save pcd_new to a new ply file:
    # o3d.io.write_point_cloud("../result/longdress_vox10_1051_hidden.ply", pcd_new, write_ascii=True)

    # # o3d.visualization.draw_geometries([pcd_new])
    # o3d.visualization.draw([pcd_new],lookat=[ 300.0, 600, 200 ],eye=camera,up=[0., 1, 0.])