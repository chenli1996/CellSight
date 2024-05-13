import torch
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import pandas as pd
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

def mask_outputs_batch_y(outputs, batch_y,output_size=1):
    output_size = 1
    mask = batch_y[:,:,:,0] != 0 # (batch_size, self.output_window, self.num_nodes)
    mask = mask.float() # (batch_size, self.output_window, self.num_nodes)
    mask = mask.unsqueeze(3) # (batch_size * self.output_window, self.num_nodes, 1)
    mask = mask.repeat(1, 1, 1, output_size) # (batch_size , self.output_window, self.num_nodes, output_size)            
    assert output_size == 1
    # batch_y is only for occlusion now otherwise, batch_y[:,:,:,0:3] or other features
    batch_y = batch_y[:,:,:,2:3] # (batch_size, self.output_window, self.num_nodes, output_size)
    # mask_b = mask
    # outputs_b = outputs
    # batch_y_b = batch_y
    # print('before',mask_b[0,100,:,0],outputs_b[0,100,:,0],batch_y_b[0,100,:,0])
    mask = mask.expand_as(outputs)
    outputs = outputs * mask
    batch_y = batch_y * mask    
    # mask_a = mask
    # outputs_a = outputs
    # batch_y_a = batch_y
    # print('after',mask_a[0,100,:,0],outputs_a[0,100,:,0],batch_y_a[0,100,:,0])
    # import pdb;pdb.set_trace()
    return outputs, batch_y

def get_val_loss(mymodel,val_loader,criterion):
    # get val_loss
    mymodel.eval()
    val_loss = 0
    with torch.no_grad():
        iter2 = 0
        for i,(batch_x, batch_y) in enumerate (val_loader):
            if torch.cuda.is_available():
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
            else:
                batch_x=batch_x
                batch_y=batch_y
            outputs = mymodel(batch_x)
            outputs,batch_y = mask_outputs_batch_y(outputs, batch_y)                
            loss = criterion(outputs,batch_y)
            val_loss += loss.item()
            iter2+=1
    val_loss = val_loss/iter2
    return val_loss   

def getedge(x):
    # df = pd.read_csv(x, nrows=edge_number)
    df = pd.read_csv(x)
    # import pdb;pdb.set_trace()
    # get the df where edge_feature is 1
    # df = df[df['edge_feature']==1]
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
    # x = df[data_type[2]].to_numpy()
    # x_n = (x - min(x)) / (max(x) - min(x))

    # x_2= df[data_type[1]].to_numpy()

    x_list = []
    for feature in data_type:
        x_2 = df[feature].to_numpy()
        x_list.append((x_2 - min(x_2)) / (max(x_2) - min(x_2)))
        # x_list.append(x_2)


    #x_3= df[data_type[2]].to_numpy()
    # return x_n,x_2
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
    for i in range(len(data)-history-future):
       data_x.append(data[i:i+history])
       data_y.append(data[i+history:i+history+future])
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    # data_y only get the part of feature, from shape (number of sample, 3, num_nodes, 7)
    # data_y = data_y[:,:,:,2:3]#only occlusion feature

    return data_x,data_y


def get_train_test_data_on_users_all_videos(history,future,p_start=1,p_end=28,voxel_size=128,num_nodes=240):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
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
    if os.path.exists(f'./data/data/all_videos_train_x{history}_{future}.npy'):
        print('load data from file')
        # add future history in the file name
        # add new directory data/data
        train_x = np.load(f'./data/data/all_videos_train_x{history}_{future}.npy')
        train_y = np.load(f'./data/data/all_videos_train_y{history}_{future}.npy')
        test_x = np.load(f'./data/data/all_videos_test_x{history}_{future}.npy')
        test_y = np.load(f'./data/data/all_videos_test_y{history}_{future}.npy')
        val_x = np.load(f'./data/data/all_videos_val_x{history}_{future}.npy')
        val_y = np.load(f'./data/data/all_videos_val_y{history}_{future}.npy')        
    else:
        print('generate data from files')
        train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        
        # save data to file with prefix is all_videos
        np.save(f'./data/data/all_videos_train_x{history}_{future}.npy',train_x)
        np.save(f'./data/data/all_videos_train_y{history}_{future}.npy',train_y)
        np.save(f'./data/data/all_videos_test_x{history}_{future}.npy',test_x)
        np.save(f'./data/data/all_videos_test_y{history}_{future}.npy',test_y)
        np.save(f'./data/data/all_videos_val_x{history}_{future}.npy',val_x)
        np.save(f'./data/data/all_videos_val_y{history}_{future}.npy',val_y)
        print('data saved')
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)
    return train_x,train_y,test_x,test_y,val_x,val_y

# # participant = 'P01_V1'
# def get_train_test_split_user():
#     train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
#     for user_i in tqdm(range(1,16)):
#         participant = 'P'+str(user_i).zfill(2)+'_V1'
#         # generate graph voxel grid features
#         prefix = f'{pcd_name}_VS{voxel_size}'
#         node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
#         column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
#         # column_name ['occlusion_feature']
#         norm_data=getdata_normalize(node_feature_path,column_name)
#         # x=np.array(list(zip(a1)))
#         # x=np.array(list(zip(a2)))
#         x=np.array(norm_data)
#         # x=x.reshape(1440,301,1)
#         feature_num = len(column_name)
#         # feature_num = 1
#         print('feature_num:',feature_num)
#         x=x.reshape(feature_num,-1,num_nodes)
#         # import pdb;pdb.set_trace()
#         x=x.transpose(1,2,0)
#         train_x1,train_y1,test_x1,test_y1,val_x1,val_y1=get_train_data_splituser(x,history,future)
#         train_x.append(train_x1)
#         train_y.append(train_y1)
#         test_x.append(test_x1)
#         test_y.append(test_y1)
#         val_x.append(val_x1)
#         val_y.append(val_y1)
#     train_x = np.concatenate(train_x)
#     train_y = np.concatenate(train_y)
#     test_x = np.concatenate(test_x)
#     test_y = np.concatenate(test_y)
#     val_x = np.concatenate(val_x)
#     val_y = np.concatenate(val_y)
#     return train_x,train_y,test_x,test_y,val_x,val_y
# # import pdb;pdb.set_trace()

# def get_train_test_data_on_users(history,future):
#     # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
#     # train_start = 1
#     # train_end = 5
#     # test_start = 21
#     # test_end = 26 -3
#     # val_start = 27
#     # val_end = 28
#     column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
#     # column_name ['occlusion_feature']
#     def get_train_test_data(train_start,train_end):
#         train_x,train_y = [],[]
#         for user_i in tqdm(range(train_start,train_end)):
#             participant = 'P'+str(user_i).zfill(2)+'_V1'
#             # generate graph voxel grid features
#             prefix = f'{pcd_name}_VS{voxel_size}'
#             node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
#             norm_data=getdata_normalize(node_feature_path,column_name)
#             x=np.array(norm_data)
#             feature_num = len(column_name)
#             # feature_num = 1
#             print('feature_num:',feature_num)
#             x=x.reshape(feature_num,-1,num_nodes)
#             # import pdb;pdb.set_trace()
#             x=x.transpose(1,2,0)
#             train_x1,train_y1=get_history_future_data(x,history,future)
#             if len(train_x1) == 0:
#                 print(f'no enough data{participant}')
#                 continue
#             train_x.append(train_x1)
#             train_y.append(train_y1)
#         # import pdb;pdb.set_trace()
#         # try:
#         if len(train_x) == 0:
#             return [],[]
#         train_x = np.concatenate(train_x)
#         # except:
#             # import pdb;pdb.set_trace()
#         train_y = np.concatenate(train_y)
#         return train_x,train_y
    
#     train_x,train_y = get_train_test_data(train_start,train_end)
#     test_x,test_y = get_train_test_data(test_start,test_end)
#     val_x,val_y = get_train_test_data(val_start,val_end)
#     return train_x,train_y,test_x,test_y,val_x,val_y
