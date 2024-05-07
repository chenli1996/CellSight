from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torch_geometric.nn import GATConv
from torch_geometric.data import Data,Batch
from tqdm import tqdm
from time import time
import os
from utils_graphgru import *
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

torch.set_default_tensor_type(torch.FloatTensor)
def get_train_test_data_on_users_all_videos_LR(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    # pcd_name_list = ['longdress','loot','redandblack','soldier']
    pcd_name_list = ['soldier']
    # column_name ['occlusion_feature']
    def get_train_test_data(pcd_name_list,p_start=1,p_end=28):
        # p_start = p_start + start_bias
        # p_end = p_end + end_bias
        print(f'{pcd_name_list}',f'p_start:{p_start},p_end:{p_end}')
        train_x,train_y = [],[]
        for pcd_name in pcd_name_list:
            print(f'pcd_name:{pcd_name}')
            for user_i in tqdm(range(p_start,p_end)):
                participant = 'P'+str(user_i).zfill(2)+'_V1'
                # generate graph voxel grid features
                prefix = f'{pcd_name}_VS{voxel_size}_LR'
                node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data(x,history,future)
                if len(train_x1) == 0:
                    print(f'no enough data{participant}')
                    continue
                train_x.append(train_x1)
                train_y.append(train_y1)
        # import pdb;pdb.set_trace()
        # try:
        if len(train_x) == 0:
            return [],[]
        train_x = np.concatenate(train_x)
        # except:
        # import pdb;pdb.set_trace()
        train_y = np.concatenate(train_y)
        return train_x,train_y
    # if data is saved, load it
    if os.path.exists(f'./data/data/all_videos_test_x{history}_{future}_LR.npy'):
        print('load data from file')
        # add future history in the file name
        # add new directory data/data
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_LR.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_LR.npy')
        # val_x = np.load(f'./data/data/all_videos_val_x{history}_{future}_LR.npy')
        # val_y = np.load(f'./data/data/all_videos_val_y{history}_{future}_LR.npy')       
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[0],p_start=p_start,p_end=int(p_end/2)+1)
        # val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        
        # save data to file with prefix is all_videos
        # np.save(f'./data/data/all_videos_train_x{history}_{future}.npy',train_x)
        # np.save(f'./data/data/all_videos_train_y{history}_{future}.npy',train_y)
        np.save(f'./data/data/all_videos_test_x{history}_{future}_LR.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_LR.npy',test_y)
        # np.save(f'./data/data/all_videos_val_x{history}_{future}.npy',val_x)
        # np.save(f'./data/data/all_videos_val_y{history}_{future}.npy',val_y)
        print('data saved')
    # train_x = train_x.astype(np.float32)
    # train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    # val_x = val_x.astype(np.float32)
    # val_y = val_y.astype(np.float32)
    return test_x,test_y

voxel_size = int(128)
num_nodes = 240
history,future=90,30
p_start = 1
p_end = 28
output_size = 1
train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
print('shape of train_x:',train_x.shape,'shape of train_y:',train_y.shape,
          'shape of test_x:',test_x.shape,'shape of test_y:',test_y.shape,
          'shape of val_x:',val_x.shape,'shape of val_y:',val_y.shape)
test_x_LR,test_y_LR = get_train_test_data_on_users_all_videos_LR(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
print('shape of test_x_LR:',test_x_LR.shape,'shape of test_y_LR:',test_y_LR.shape)

# test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
test_y_LR = torch.from_numpy(test_y_LR)
# get mse mae loss for LR on test data
mae = MeanAbsoluteError()
mape=MeanAbsolutePercentageError()
mse=MeanSquaredError()
if torch.cuda.is_available():
    mae = mae.to('cuda')
    mape = mape.to('cuda')
    mse = mse.to('cuda')
    test_y = test_y.to('cuda')
    test_y_LR = test_y_LR.to('cuda')
import pdb;pdb.set_trace()
for u in range(future):
    pass
    # mae.update(test_y[:,u,:,:],test_y_LR[:,u])
    # mape.update(test_y[:,u],test_y_LR[:,u])
    # mse.update(test_y[:,u],test_y_LR[:,u])





