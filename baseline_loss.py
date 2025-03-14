from operator import index
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from torch_geometric.nn import GATConv
from torch_geometric.data import Data,Batch
from tqdm import tqdm
from time import time
import os
from utils_graphgru import *
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import r2_score
torch.set_default_dtype(torch.float32)

def get_train_test_data_on_users_all_videos_LR(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):

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

        if len(train_x) == 0:
            return [],[]
        train_x = np.concatenate(train_x)

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

def get_train_test_data_on_users_all_videos_baseline(baseline,history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):

    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','theta_feature','f_theta_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
    # pcd_name_list = ['soldier']
    # column_name ['occlusion_feature']
    
    def get_train_test_data(pcd_name_list,p_start=1,p_end=28):
        # p_start = p_start + start_bias
        # p_end = p_end + end_bias
        index_end = 0
        test_index = 0
        print(f'{pcd_name_list}',f'p_start:{p_start},p_end:{p_end}')
        train_x,train_y = [],[]
        for pcd_name in pcd_name_list:
            print(f'pcd_name:{pcd_name}')
            for user_i in tqdm(range(p_start,p_end)):
                participant = 'P'+str(user_i).zfill(2)+'_V1'
                # generate graph voxel grid features
                prefix = f'{pcd_name}_VS{voxel_size}_{baseline}'
                node_feature_path = f'./data/{prefix}/{participant}node_feature_angular{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data)
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
                # train_x1,train_y1,index_end,test_index=get_history_future_data_full_index(x,history,future,index_end,test_index)
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

    if os.path.exists(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy'):
        print('load baseline data from file angular')
        # add future history in the file name
        # add new directory data/data
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_{baseline}_angular.npy')
    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        # val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_{baseline}_angular.npy',test_y)
        print('data saved')
    return test_x,test_y

def get_train_test_data_on_users_all_videos_LSTM(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):

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

def get_train_test_data_on_users_all_videos_fsvvd_baseline(baseline,dataset,history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):

    train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # initialize as np array
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','theta_feature','f_theta_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['Chatting','Pulling_trolley','News_interviewing','Sweep']
    user_list = ['ChenYongting','GuoYushan','Guozhaonian','HKY','RenZhichen','Sunqiran','WangYan','fupingyu','huangrenyi','liuxuya','sulehan','yuchen']
    # column_name ['occlusion_feature']
    def get_train_test_data(pcd_name_list,p_start=0,p_end=11):
        # p_start = p_start + start_bias
        # p_end = p_end + end_bias
        print(f'{pcd_name_list}',f'p_start:{p_start},p_end:{p_end}')
        train_x,train_y = [],[]
        index_end = 0
        test_index = 0
        for pcd_name in pcd_name_list:
            print(f'pcd_name:{pcd_name}')
            for user_i in tqdm(range(p_start,p_end)):
                participant = user_list[user_i]
                # generate graph voxel grid features
                prefix = f'{pcd_name}_VS{voxel_size}_{baseline}'
                node_feature_path = f'./data/{prefix}/{participant}node_feature_angular{history}{future}.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data,dtype=np.float32)

                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
                # train_x1,train_y1,index_end,test_index=get_history_future_data_full_index(x,history,future,index_end,test_index)
                if len(train_x1) == 0:
                    print(f'no enough data{participant}',flush=True)
                    continue
                train_x.append(train_x1)
                train_y.append(train_y1)
        # Force garbage collection
        gc.collect()

        if len(train_x) == 0:
            return np.array([]),np.array([])
        train_x = np.concatenate(train_x)

        train_y = np.concatenate(train_y)
        return train_x,train_y

    if not os.path.exists(f'./data/{dataset}'):
        os.makedirs(f'./data/{dataset}')

    if os.path.exists(f'./data/{dataset}/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy'):
        print('load baseline data from file angular')
        test_x = np.load(f'./data/{dataset}/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy')
        test_y = np.load(f'./data/{dataset}/all_videos_test_y{history}_{future}_{voxel_size}_{baseline}_angular.npy')

    else:
        print('generate data from files')
        # train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)

        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)

        np.save(f'./data/{dataset}/all_videos_test_x{history}_{future}_{voxel_size}_{baseline}_angular.npy',test_x)
        np.save(f'./data/{dataset}/all_videos_test_y{history}_{future}_{voxel_size}_{baseline}_angular.npy',test_y)


        print('data saved')

    return test_x,test_y

def compute_r2_per_sample(y_true, y_pred):
    # Compute mean of true values per sample
    y_true_mean = y_true.mean(dim=1, keepdim=True)

    # Compute total sum of squares (SS_tot) per sample
    ss_tot = ((y_true - y_true_mean) ** 2).sum(dim=1)

    # Compute residual sum of squares (SS_res) per sample
    ss_res = ((y_true - y_pred) ** 2).sum(dim=1)

    # Compute R² score per sample
    r2_scores = torch.ones_like(ss_tot)
    non_zero_tot = ss_tot != 0
    r2_scores[non_zero_tot] = 1 - ss_res[non_zero_tot] / ss_tot[non_zero_tot]
    r2_scores[~non_zero_tot] = 0.0  # Set R² to 0.0 when SS_tot is zero

    return r2_scores

def baseline_loss_eval(dataset,baseline,predict_end_index,history):

    # predict_end_index = 4 # 2 infov, 3 visibility, 4 resolution

    print(f'dataset:{dataset}, baseline:{baseline}, predict_end_index:{predict_end_index}')
    if dataset == 'fsvvd_raw':
        voxel_size = 0.6
        p_start = 0
        p_end = 11
        edge_prefix = str(voxel_size) + 'fsvvd_raw'
    elif dataset == 'fsvvd_filtered':
        voxel_size = 0.4
        p_start = 0
        p_end = 11
        edge_prefix = str(voxel_size) + 'fsvvd_filtered'
    elif dataset == '8i':
        voxel_size = 128
        p_start = 1
        p_end = 28
        edge_prefix = str(voxel_size)

    if voxel_size == 128:
        num_nodes = 240
    elif voxel_size == 64:
        num_nodes = 1728
    elif voxel_size == 0.6: # fsvvd full raw
        num_nodes = 280
    # elif voxel_size == 0.4:
    else:
        num_nodes = None



    output_size = 1

    mse_list = []
    R2_score_list = []


    # for future in [1,10,30,60,150]:
    for future in [1]:
        print(f'history:{history},future:{future}')
        output_size = 1
        # load ground truth data
        if dataset == '8i':
            train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
        else:
            train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos_fsvvd(dataset,history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
        print('shape of train_x:',train_x.shape,'shape of train_y:',train_y.shape)
        del train_x,train_y,val_x,val_y,test_x

        if dataset == '8i':
             test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_baseline(baseline,history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
        else:
            test_x_TLR,test_y_TLR = get_train_test_data_on_users_all_videos_fsvvd_baseline(baseline,dataset,history,future,p_start,p_end,voxel_size,num_nodes)
        del test_x_TLR
        print('shape of test_y:',test_y.shape,'shape of test_y_TLR:',test_y_TLR.shape)
        assert test_y.shape == test_y_TLR.shape
        # import pdb;pdb.set_trace()
        test_y = torch.from_numpy(test_y)
        test_y_TLR = torch.from_numpy(test_y_TLR)
        # import pdb;pdb.set_trace()



        # get mse mae loss for LR on test data
        mae = MeanAbsoluteError()
        mape=MeanAbsolutePercentageError()
        mse=MeanSquaredError()
        # r2_score = R2Score()
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
        for index in range(0,test_y.size(0),1):
            baseline_output = test_y_TLR[:, u, :, predict_end_index-output_size:predict_end_index].contiguous()[index].squeeze(-1)
            gt_output = test_y[:, u, :, predict_end_index-output_size:predict_end_index].contiguous()[index].squeeze(-1)
            mse_temp = mse(baseline_output, gt_output).cpu().detach().numpy()
            mae_temp = mae(baseline_output, gt_output).cpu().detach().numpy()


            if index == 1635: #fsvvd p4
                print(f'index:{index},MSE:{mse_temp}')
                print('baseline:',baseline_output)
                print('gt:',gt_output)
            


        y_true = test_y[:, u, :, predict_end_index - output_size : predict_end_index].contiguous().squeeze(-1)
        y_pred = test_y_TLR[:, u, :, predict_end_index - output_size : predict_end_index].contiguous().squeeze(-1)
        squared_errors = (y_true - y_pred) ** 2
        std_squared_errors = torch.std(squared_errors)
        print("Standard Deviation of Squared Errors:", std_squared_errors.item())
        final_r2_score = r2_score(y_true.cpu().view(-1), y_pred.cpu().view(-1))
        pred_y_o_non_zero,test_y_o_non_zero = mask_outputs_batch_y_cut(test_y_TLR,test_y, u, output_size, predict_end_index)
        mse_non_zero = mse(pred_y_o_non_zero, test_y_o_non_zero).cpu().detach().numpy()
        print(f'MSE_non_zero-o:{mse_non_zero}')
        std_squared_errors_non_zero = torch.std((pred_y_o_non_zero - test_y_o_non_zero) ** 2)
        print(f'std_squared_errors_non_zero:{std_squared_errors_non_zero}')
        r2_score_non_zero = r2_score(test_y_o_non_zero.cpu(), pred_y_o_non_zero.cpu())
        print(f'R² Score (Non-Zero): {r2_score_non_zero}')


        # y_true_non_zero is the true values that are not zero
        y_true_non_zero = y_true[y_true != 0]
        y_pred_non_zero = y_pred[y_true != 0]
        squared_errors_non_zero = (y_true_non_zero - y_pred_non_zero) ** 2
        std_squared_errors_non_zero = torch.std(squared_errors_non_zero)
        mse_non_zero = mse(y_pred_non_zero, y_true_non_zero).cpu().detach().numpy()
        print(f'std_squared_errors_non_zero:{std_squared_errors_non_zero}')
        print(f'mse_non_zero-v:{mse_non_zero}')


        MAE_d = mae(test_y[:,u,:,predict_end_index-output_size:predict_end_index],test_y_TLR[:,u,:,predict_end_index-output_size:predict_end_index]).cpu().detach().numpy()
        print(f'MSE:{MSE_d},MAE:{MAE_d}, R² Score: {final_r2_score}',f'history:{history},future:{future}')
        # import pdb;pdb.set_trace()
        mse_list.append(round(MSE_d.item(),4))
        R2_score_list.append(round(final_r2_score,3))


        del test_y,test_y_TLR
    print(f'dataset:{dataset},baseline:{baseline},predict_end_index:{predict_end_index}')
    print(f'{baseline}:MSE_list:{mse_list},R2_score_list:{R2_score_list}')




if __name__ == '__main__':
    for dataset in ['8i']:
    # for dataset in ['fsvvd_raw']:
        # for baseline in ['LSTM','MLP','TLR','LR']:
        for baseline in ['LSTM']:
            # for predict_end_index in [2,3,4]:
            for predict_end_index in [4]:
                baseline_loss_eval(dataset,baseline,predict_end_index,history=90)
    
    # # LR for 30 history
    # for dataset in ['8i']:
    #     for baseline in ['MLP']:
    #         for predict_end_index in [4]:
    #             baseline_loss_eval(dataset,baseline,predict_end_index,history=30)




