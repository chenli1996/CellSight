# from examples import test
from operator import index
import torch
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gc

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, val_loss_min =float('inf'), path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        if val_loss_min == float('inf'):
            self.best_score = None
        else:
            self.best_score = -val_loss_min
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        # if score is nan, return
        if np.isnan(score):
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        # check whether path exists, if not, create path
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def mask_outputs_batch_y(outputs, batch_y,output_size,predict_index_end):
    output_size = output_size
    assert output_size == 1 or output_size == 2
    mask = batch_y[:,:,:,0] != 0 # (batch_size, self.output_window, self.num_nodes)
    mask = mask.float() # (batch_size, self.output_window, self.num_nodes)
    mask = mask.unsqueeze(3) # (batch_size * self.output_window, self.num_nodes, 1)
    mask = mask.repeat(1, 1, 1, output_size) # (batch_size , self.output_window, self.num_nodes, output_size)            
    # assert output_size == 1
    # batch_y is only for occlusion now otherwise, batch_y[:,:,:,0:3] or other features
    batch_y = batch_y[:,:,:,predict_index_end-output_size:predict_index_end] # (batch_size, self.output_window, self.num_nodes, output_size)
    mask = mask.expand_as(outputs)
    outputs = outputs * mask
    batch_y = batch_y * mask    
    return outputs, batch_y

def mask_outputs_batch_y_cut(outputs, batch_y,u,output_size,predict_index_end):
    output_size = output_size
    assert output_size == 1 or output_size == 2
    mask = batch_y[:,u,:,0] != 0 # (batch_size, self.output_window, self.num_nodes)
    # mask = mask.float() # (batch_size, self.output_window, self.num_nodes)
    mask = mask.unsqueeze(-1) # (batch_size * self.output_window, self.num_nodes, 1)
    batch_y = batch_y[:,u,:,predict_index_end-output_size:predict_index_end] # (batch_size, self.output_window, self.num_nodes, output_size)
    outputs = outputs[:,u,:,predict_index_end-output_size:predict_index_end]
    # import pdb;pdb.set_trace()
    mask = mask.expand_as(outputs)
    outputs = outputs[mask]
    batch_y = batch_y[mask]

    return outputs, batch_y

def get_val_loss(mymodel,val_loader,criterion,output_size,target_output=1,predict_index_end=3,object_driven=False):
    # get val_loss
    mymodel.eval()
    val_loss = 0
    with torch.no_grad():
        iter2 = 0
        for i,(batch_x, batch_y) in enumerate (val_loader):
            batch_y_object = batch_y.clone()
            batch_y=batch_y[:,-target_output,:,:]
            # keep batch_y as (batch_size, 1, self.num_nodes, self.output_dim)
            batch_y = batch_y.unsqueeze(1)  

            if torch.cuda.is_available():
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
                batch_y_object = batch_y_object.cuda()
            if object_driven:
                outputs = mymodel.forward_output1_o(batch_x,batch_y_object)
            else:
                outputs = mymodel(batch_x)
            #  -------------            
            if predict_index_end in [3,4]:
                # occlusion prediction, we have object driven and need mask
                outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size,predict_index_end=predict_index_end)
            elif predict_index_end == 5:
                batch_y = batch_y[:,:,:,3-output_size:3]
            elif predict_index_end == 2:
            # outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size,predict_index_end=3) 
                batch_y = batch_y[:,:,:,predict_index_end-output_size:predict_index_end]               
            else:
                print('predict_index_end not in [2,3,4,5]')

            # -------------
            loss = criterion(outputs,batch_y)
            val_loss += loss.item()
            iter2+=1
    val_loss = val_loss/iter2
    return val_loss   

def getedge(x):
    df = pd.read_csv(x)
    r1 = df.loc[:, 'start_node'].values
    r2 = df.loc[:, 'end_node'].values
    return r1, r2
def save(x,y,z,real,prediction,history):

    x=x.cpu().numpy()
    y=y.cpu().numpy()
    z=z.detach().cpu().numpy()
    history.append(x)
    real.append(y)
    prediction.append(z)
    return real,prediction,history

def getdata(file_name,data_type):
    df = pd.read_csv(file_name)
    x=df.loc[df.node==0]
    x = df[data_type].to_numpy()
    return x

def getdata_normalize(file_name,data_type):
    df = pd.read_csv(file_name)

    x_list = []
    for feature in data_type:
        x_2 = df[feature].to_numpy()
        x_list.append((x_2 - min(x_2)) / (max(x_2) - min(x_2)))
    return x_list

def getdata_nn(file_name,data_type):
    df = pd.read_csv(file_name)
    x_list = []
    for feature in data_type:
        x_2 = df[feature].to_numpy()
        x_list.append(x_2)
    return x_list

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def get_train_data_splituser(data,history,future):
    data_x = []
    data_y = []
    for i in range(len(data)-history-future):
       data_x.append(data[i:i+history])
       data_y.append(data[i+history:i+history+future])
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    assert data_y.shape[0] >= future + history
    assert data_x.shape[0] >= future + history
    # data_y only get the part of feature, from shape (number of sample, 3, num_nodes, 7)
    # data_y = data_y[:,:,:,2:3]#only occlusion feature


    size1 = int(len(data_x) * 0.8)
    size2 = int(len(data_x) * 1)
    train_x = data_x[:size1]
    train_y = data_y[:size1]
    test_x = data_x[size1:size2]
    test_y = data_y[size1:size2]
    val_x = data_x[size2:]
    val_y = data_y[size2:]
    return train_x,train_y,test_x,test_y,val_x,val_y

def get_history_future_data(data,history,future):
    data_x = []
    data_y = []
    if data.shape[0] <= future + history:
        return data_x,data_y
    i = 0
    while i < len(data)-history-future:
        data_x.append(data[i:i+history])
        data_y.append(data[i+history:i+history+future])
        if (i+history+future) % 150 == 0:
            i += 150
        else:
            i += 1
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    # data_y only get the part of feature, from shape (number of sample, 3, num_nodes, 7)
    # data_y = data_y[:,:,:,2:3]#only occlusion feature

    return data_x,data_y

def get_history_future_data_full(data,history,future):
    data_x = []
    data_y = []
    if data.shape[0] <= future + history:
        return data_x,data_y
    i = 0
    while i < len(data)-history-future:
        data_x.append(data[i:i+history])
        data_y.append(data[i+history:i+history+future])
        i += 1

    return data_x,data_y

def get_history_future_data_full_index(data,history,future,index_end,test_index):
    data_x = []
    data_y = []
    if data.shape[0] <= future + history:
        return data_x,data_y
    i = 0
    while i < len(data)-history-future:
        data_x.append(data[i:i+history])
        data_y.append(data[i+history:i+history+future])
        index_end = i+history+future
        print(f'index num{index_end},test_index{test_index},frame_index{(index_end)%150*2}')
        i += 1
        test_index += 1
    return data_x,data_y,index_end,test_index

def get_train_test_data_on_users_all_videos(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # initialize as np array
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','theta_feature','f_theta_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    pcd_name_list = ['longdress','loot','redandblack','soldier']
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
                prefix = f'{pcd_name}_VS{voxel_size}'
                node_feature_path = f'./data/{prefix}/{participant}node_feature_angular.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data,dtype=np.float32)
                # read the data from csv file as numpy array
                feature_num = len(column_name)
                x=x.reshape(feature_num,-1,num_nodes)
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
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
    if os.path.exists(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy'):
        print('load GT data from file angular')
        # add future history in the file name
        # add new directory data/data
        train_x = np.load(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy')
        train_y = np.load(f'./data/data/all_videos_train_y{history}_{future}_{voxel_size}_angular.npy')
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_angular.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_angular.npy')
        val_x = np.load(f'./data/data/all_videos_val_x{history}_{future}_{voxel_size}_angular.npy')
        val_y = np.load(f'./data/data/all_videos_val_y{history}_{future}_{voxel_size}_angular.npy')

    else:
        print('generate data from files')
        train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        val_x = val_x.astype(np.float32)
        val_y = val_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy',train_x)
        np.save(f'./data/data/all_videos_train_y{history}_{future}_{voxel_size}_angular.npy',train_y)
        np.save(f'./data/data/all_videos_test_x{history}_{future}_{voxel_size}_angular.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}_{voxel_size}_angular.npy',test_y)
        np.save(f'./data/data/all_videos_val_x{history}_{future}_{voxel_size}_angular.npy',val_x)
        np.save(f'./data/data/all_videos_val_y{history}_{future}_{voxel_size}_angular.npy',val_y)

        print('data saved')

    return train_x,train_y,test_x,test_y,val_x,val_y


def get_train_test_data_on_users_all_videos_fsvvd(dataset,history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
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
        print(f'{pcd_name_list}',f'range p_start:{p_start},p_end:{p_end}')
        train_x,train_y = [],[]
        for pcd_name in pcd_name_list:
            print(f'pcd_name:{pcd_name}')
            for user_i in tqdm(range(p_start,p_end)):
                # participant = 'P'+str(user_i).zfill(2)+'_V1'
                participant = user_list[user_i]
                # generate graph voxel grid features
                prefix = f'{pcd_name}_VS{voxel_size}'
                # node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
                node_feature_path = f'./data/{prefix}/{participant}node_feature_angular.csv'
                norm_data=getdata_normalize(node_feature_path,column_name)
                x=np.array(norm_data,dtype=np.float32)
                # read the data from csv file as numpy array
                feature_num = len(column_name)
                # feature_num = 1
                # print('feature_num:',feature_num)
                x=x.reshape(feature_num,-1,num_nodes)
                # import pdb;pdb.set_trace()
                x=x.transpose(1,2,0)
                train_x1,train_y1=get_history_future_data_full(x,history,future)
                if len(train_x1) == 0:
                    print(f'no enough data{participant}',flush=True)
                    continue
                train_x.append(train_x1)
                train_y.append(train_y1)
        # Force garbage collection
        gc.collect()
        # import pdb;pdb.set_trace()
        # try:
        if len(train_x) == 0:
            return np.array([]),np.array([])
        train_x = np.concatenate(train_x)
        # except:
        # import pdb;pdb.set_trace()
        train_y = np.concatenate(train_y)
        return train_x,train_y
    if not os.path.exists(f'./data/{dataset}'):
        os.makedirs(f'./data/{dataset}')

    if os.path.exists(f'./data/{dataset}/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy'):
        print('load training data from file angular')
        # add future history in the file name
        # add new directory data/data
        train_x = np.load(f'./data/{dataset}/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy')
        train_y = np.load(f'./data/{dataset}/all_videos_train_y{history}_{future}_{voxel_size}_angular.npy')
        test_x = np.load(f'./data/{dataset}/all_videos_test_x{history}_{future}_{voxel_size}_angular.npy')
        test_y = np.load(f'./data/{dataset}/all_videos_test_y{history}_{future}_{voxel_size}_angular.npy')
        val_x = np.load(f'./data/{dataset}/all_videos_val_x{history}_{future}_{voxel_size}_angular.npy')
        val_y = np.load(f'./data/{dataset}/all_videos_val_y{history}_{future}_{voxel_size}_angular.npy')



    else:
        print('generate data from files')
        train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end+1) # all 12 users
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1) # first 6 users
        val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end+1) # last 6 users
        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)
        val_x = val_x.astype(np.float32)
        val_y = val_y.astype(np.float32)
        # save data to file with prefix is all_videos
        np.save(f'./data/{dataset}/all_videos_train_x{history}_{future}_{voxel_size}_angular.npy',train_x)
        np.save(f'./data/{dataset}/all_videos_train_y{history}_{future}_{voxel_size}_angular.npy',train_y)
        np.save(f'./data/{dataset}/all_videos_test_x{history}_{future}_{voxel_size}_angular.npy',test_x)
        np.save(f'./data/{dataset}/all_videos_test_y{history}_{future}_{voxel_size}_angular.npy',test_y)
        np.save(f'./data/{dataset}/all_videos_val_x{history}_{future}_{voxel_size}_angular.npy',val_x)
        np.save(f'./data/{dataset}/all_videos_val_y{history}_{future}_{voxel_size}_angular.npy',val_y)


        print('data saved')

    return train_x,train_y,test_x,test_y,val_x,val_y
