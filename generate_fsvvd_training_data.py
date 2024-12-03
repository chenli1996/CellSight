from utils_graphgru import get_train_test_data_on_users_all_videos_fsvvd

# need 256G memory to generate the data
for future in [150,60,30,10,1]:
    print('future:',future)
    dataset = '8i'
    history = 30
    voxel_size = 0.6
    p_start = 0
    p_end = 11
    num_nodes = 280
    train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos_fsvvd(dataset,history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)