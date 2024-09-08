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
torch.set_default_dtype(torch.float32)

def get_train_test_data_on_users_all_videos_LR(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
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
                node_feature_path = f'./data/{prefix}/{participant}node_feature{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
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
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_LR.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_LR.npy')
        # val_x = np.load(f'./data/data/all_videos_val_x{history}_{future}_LR.npy')
        # val_y = np.load(f'./data/data/all_videos_val_y{history}_{future}_LR.npy')       
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
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

def get_train_test_data_on_users_all_videos_TLR(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
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
                prefix = f'{pcd_name}_VS{voxel_size}_TLR_per'
                node_feature_path = f'./data/{prefix}/{participant}node_feature{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
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
    # clip = 600
    # print('clip:',clip)
    # if os.path.exists(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_{clip}.npy'):
    if os.path.exists(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_TLR.npy'):
        print('load data from file')
        # add future history in the file name
        # add new directory data/data
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_TLR.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_TLR.npy')
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        # val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_TLR.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_TLR.npy',test_y)
        print('data saved')
    return test_x,test_y

def get_train_test_data_on_users_all_videos_MLP(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
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
                prefix = f'{pcd_name}_VS{voxel_size}_MLP'
                node_feature_path = f'./data/{prefix}/{participant}node_feature{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
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
    # clip = 600
    # print('clip:',clip)
    # if os.path.exists(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_{clip}.npy'):
    if os.path.exists(f'./data/data/ddall_videos_test_x{history}_{future}_{voxel_size}_MLP.npy'):
        print('load data from file')
        # add future history in the file name
        # add new directory data/data
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_MLP.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_MLP.npy')
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        # val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_MLP.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_MLP.npy',test_y)
        print('data saved')
    return test_x,test_y

def get_train_test_data_on_users_all_videos_LSTM(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
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
                prefix = f'{pcd_name}_VS{voxel_size}_LSTM'
                node_feature_path = f'./data/{prefix}/{participant}node_feature{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
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
    # clip = 600
    # print('clip:',clip)
    # if os.path.exists(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_{clip}.npy'):
    if os.path.exists(f'./data/data/ddall_videos_test_x{history}_{future}_{voxel_size}_LSTM.npy'):
        print('load data from file')
        # add future history in the file name
        # add new directory data/data
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_LSTM.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_LSTM.npy')
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        # val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_LSTM.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_LSTM.npy',test_y)
        print('data saved')
    return test_x,test_y




voxel_size = int(128)
if voxel_size == 128:
    num_nodes = 240
elif voxel_size == 64:
    num_nodes = 1728
else:
    num_nodes = None
# history = 10
# for future in [1,10,30,60]:
history=90
# for future in [1,10,30,60]:
# history = 30
predict_end_index = 2 #infov
# predict_end_index = 3 #visibility
output_size = 1

# for future in [30]:
for future in [10,30,60,150]:
    print(f'history:{history},future:{future}')
    p_start = 1
    p_end = 28
    output_size = 1
    train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)

    del train_x,train_y,val_x,val_y,test_x
    # test_x_LR,test_y_LR = get_train_test_data_on_users_all_videos_LR(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    # del test_x_LR
    # print('shape of test_y:',test_y.shape,'shape of test_y_LR:',test_y_LR.shape)
    # test_y = torch.from_numpy(test_y)
    # test_y_LR = torch.from_numpy(test_y_LR)

    # test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_TLR(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    # test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_MLP(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    # test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_LR(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_LSTM(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    del test_x_TLR
    print('shape of test_y:',test_y.shape,'shape of test_y_TLR:',test_y_TLR.shape)
    test_y = torch.from_numpy(test_y)
    test_y_TLR = torch.from_numpy(test_y_TLR)



    # get mse mae loss for LR on test data
    mae = MeanAbsoluteError()
    mape=MeanAbsolutePercentageError()
    mse=MeanSquaredError()
    if torch.cuda.is_available():
        mae = mae.to('cuda')
        mape = mape.to('cuda')
        mse = mse.to('cuda')
        test_y = test_y.to('cuda')
        # test_y_LR = test_y_LR.to('cuda')
        test_y_TLR = test_y_TLR.to('cuda')
    MAE_list = []
    MSE_list = []
    u=future-1
    # outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size,predict_end_index)
    # test_y_TLR,test_y = mask_outputs_batch_y(test_y_TLR, test_y,output_size,predict_end_index)
    # import pdb;pdb.set_trace()
    # get a large loss for TLR
    # for i in range(0,test_y.size(0),1):
    #     # import pdb;pdb.set_trace()
    #     if i==553:
    #         print('TLR',test_y_TLR[i, u, :, predict_end_index-output_size:predict_end_index].view(30,8))
    #         print('gt',test_y[i, u, :, predict_end_index-output_size:predict_end_index].view(30,8))
    #     MSE = mse(test_y[i, u, :, predict_end_index-output_size:predict_end_index].contiguous(), test_y_TLR[i, u, :, predict_end_index-output_size:predict_end_index].contiguous()).cpu().detach().numpy()
    #     MAE = mae(test_y[i, u,:,predict_end_index-output_size:predict_end_index],test_y_TLR[i,u,:,predict_end_index-output_size:predict_end_index]).cpu().detach().numpy()
    #     # import pdb;pdb.set_trace()
    #     if abs(MSE-0.138) < 0.1 and MAE>0.2:
    #         print(f'MSE:{MSE},MAE:{MAE}',f'index:{i}')
    
    MSE_d = mse(test_y[:, u, :, predict_end_index-output_size:predict_end_index].contiguous(), test_y_TLR[:, u, :, predict_end_index-output_size:predict_end_index].contiguous()).cpu().detach().numpy()    
    MAE_d = mae(test_y[:,u,:,predict_end_index-output_size:predict_end_index],test_y_TLR[:,u,:,predict_end_index-output_size:predict_end_index]).cpu().detach().numpy()
    print(f'MSE:{MSE_d},MAE:{MAE_d}',f'history:{history},future:{future}')
    # get the var of test_y[:,u,:,2:3] after masking off all zeros
    # test_y = test_y_TLR
    # test_y = test_y.cpu().detach().numpy()
    # test_y = test_y[:,u,:,2:3]
    # mask = test_y != 0
    # test_y = test_y[mask]
    # var = np.var(test_y)
    # # get the distrubution result of test_y
    # plt.hist(test_y.ravel(),bins=100)
    # plt.savefig(f'./data/fig/test_y_{history}_{future}.png')

    # print(f'var:{var},history:{history},future:{future}')


    del test_y,test_y_TLR









