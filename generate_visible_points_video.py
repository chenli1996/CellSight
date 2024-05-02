from point_cloud_FoV_utils import *
import open3d as o3d
print(o3d.__version__)
import cProfile
import pstats

if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # check open3d version

    for pcd_name in ['soldier']:
#    for pcd_name in ['longdress','loot','redandblack','soldier']:
        # for user in ['P02_V1']:
        for user_i in range(4,28):
            user = 'P'+str(user_i).zfill(2)+'_V1'
            positions,orientations = get_point_cloud_user_trajectory(pcd_name=pcd_name,participant=user)
            end_index = len(positions)
            # end_index = 30
            for index in range(0, end_index,1):
                print('index:',index)
                save_rendering_from_given_FoV_traces(positions,orientations,
                                        trajectory_index=index,point_cloud_name=pcd_name,user=user,prefix='',save=True,render_flag=False)
            # cd to  this directory '../result/'+point_cloud_name+'/'+user and run the following command ffmpeg -framerate 30 -pattern_type glob -i 'fov_*.png' -c:v libx264 -pix_fmt yuv420p output_video_user.mp4
                
                # get all file whose name include 'vs code' in the current directory linux

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()
                
                
