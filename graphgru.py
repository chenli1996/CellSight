#!/bin/env python
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import pickle
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

torch.set_default_tensor_type(torch.DoubleTensor)
# def getedge(x,edge_number):
#     df = pd.read_csv(x, nrows=edge_number)
#     r1 = df.loc[:, 'row'].values
#     r2 = df.loc[:, 'column'].values
#     return r1, r2
def getedge(x,edge_number):
    df = pd.read_csv(x, nrows=edge_number)
    # get the df where edge_feature is 1
    df = df[df['edge_feature']==1]
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
    data_y = data_y[:,:,:,2:3]#only occlusion feature

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
    data_y = data_y[:,:,:,2:3]#only occlusion feature

    return data_x,data_y


# participant = 'P01_V1'
def get_train_test_split_user():
    train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    for user_i in tqdm(range(1,16)):
        participant = 'P'+str(user_i).zfill(2)+'_V1'
        # generate graph voxel grid features
        prefix = f'{pcd_name}_VS{voxel_size}'
        node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
        column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
        # column_name ['occlusion_feature']
        norm_data=getdata_normalize(node_feature_path,column_name)
        # x=np.array(list(zip(a1)))
        # x=np.array(list(zip(a2)))
        x=np.array(norm_data)
        # x=x.reshape(1440,301,1)
        feature_num = len(column_name)
        # feature_num = 1
        print('feature_num:',feature_num)
        x=x.reshape(feature_num,-1,num_nodes)
        # import pdb;pdb.set_trace()
        x=x.transpose(1,2,0)
        train_x1,train_y1,test_x1,test_y1,val_x1,val_y1=get_train_data_splituser(x,history,future)
        train_x.append(train_x1)
        train_y.append(train_y1)
        test_x.append(test_x1)
        test_y.append(test_y1)
        val_x.append(val_x1)
        val_y.append(val_y1)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)
    return train_x,train_y,test_x,test_y,val_x,val_y
# import pdb;pdb.set_trace()

def get_train_test_data_on_users(history,future):
    # train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
    # train_start = 1
    # train_end = 5
    # test_start = 21
    # test_end = 26 -3
    # val_start = 27
    # val_end = 28
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    # column_name ['occlusion_feature']
    def get_train_test_data(train_start,train_end):
        train_x,train_y = [],[]
        for user_i in tqdm(range(train_start,train_end)):
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
    
    train_x,train_y = get_train_test_data(train_start,train_end)
    test_x,test_y = get_train_test_data(test_start,test_end)
    val_x,val_y = get_train_test_data(val_start,val_end)
    return train_x,train_y,test_x,test_y,val_x,val_y

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
    if os.path.exists(f'./data/all_videos_train_x{history}_{future}.npy'):
        print('load data from file')
        # add future history in the file name
        train_x = np.load(f'./data/all_videos_train_x{history}_{future}.npy')
        train_y = np.load(f'./data/all_videos_train_y{history}_{future}.npy')
        test_x = np.load(f'./data/all_videos_test_x{history}_{future}.npy')
        test_y = np.load(f'./data/all_videos_test_y{history}_{future}.npy')
        val_x = np.load(f'./data/all_videos_val_x{history}_{future}.npy')
        val_y = np.load(f'./data/all_videos_val_y{history}_{future}.npy')        
    else:
        print('generate data from files')
        train_x,train_y = get_train_test_data(pcd_name_list[0:3],p_start=p_start,p_end=p_end)
        test_x,test_y = get_train_test_data(pcd_name_list[3:],p_start=p_start,p_end=int(p_end/2)+1)
        val_x,val_y = get_train_test_data(pcd_name_list[3:],p_start=int(p_end/2)+1,p_end=p_end)
        
        # save data to file with prefix is all_videos
        np.save(f'./data/all_videos_train_x{history}_{future}.npy',train_x)
        np.save(f'./data/all_videos_train_y{history}_{future}.npy',train_y)
        np.save(f'./data/all_videos_test_x{history}_{future}.npy',test_x)
        np.save(f'./data/all_videos_test_y{history}_{future}.npy',test_y)
        np.save(f'./data/all_videos_val_x{history}_{future}.npy',val_x)
        np.save(f'./data/all_videos_val_y{history}_{future}.npy',val_y)
        print('data saved')

    return train_x,train_y,test_x,test_y,val_x,val_y

#######################################################
class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int,num_nodes: int, feature_num: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.feature_num = feature_num
        self.weights = nn.Parameter(
            torch.DoubleTensor(self._num_gru_units + self.feature_num, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()
        self.num_nodes = num_nodes

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size = hidden_state.shape[0]
        # assert batch_size == 200
        inputs = inputs.reshape((batch_size, self.num_nodes, self.feature_num))
        # inputs (batch_size, num_nodes, feature_num)
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, self.num_nodes, self._num_gru_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_gru_units + self.feature_num))
        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, self.num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        #outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class GraphGRUCell(nn.Module):
    def __init__(self, num_units, num_nodes, r1,r2, device, input_dim=1):
        super(GraphGRUCell, self).__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.device = device
        self.act = torch.tanh
        self.init_params()
        self.r1 = r1
        self.r2 = r2
        self.GRU1 = GRULinear(100, 200, self.num_nodes,self.input_dim)
        self.GRU2 = GRULinear(100, 100, self.num_nodes,self.input_dim)
        # self.GCN3 = GATConv(101, 100)
        self.GCN3 = GATConv(100+self.input_dim, 100)
    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self.device))
        bias_0 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self.device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self.device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self.device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

    def forward(self, inputs, state):
        batch_size = state.shape[0]
        state=self._gc3(state,inputs, self.num_units)
        output_size = 2 * self.num_units
        value = torch.sigmoid(
            self.GRU1(inputs, state))  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))
        c = self.act(self.GRU2(inputs, r * state))
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c
        return new_state




    def _gc3(self, state, inputs, output_size, bias_start=0.0):

        batch_size = state.shape[0]
        # assert batch_size == 200
        # import pdb;pdb.set_trace()

        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))
        inputs_and_state = torch.cat([state, inputs], dim=2)
        input_size = inputs_and_state.shape[2]
        x = inputs_and_state.to(self.device)
        # edge_index = torch.tensor([self.r1, self.r2], dtype=torch.long).to(self.device)
        edge_index = torch.tensor(np.stack((np.array(self.r1),np.array(self.r2))), dtype=torch.long).to(self.device)
        # import pdb;pdb.set_trace()
        b=[]
        # for i in x:
        #   x111=Data(x=i,edge_index=edge_index)
        #   xx=self.GCN3(x111.x,x111.edge_index)
        #   b.append(xx)
        # x1=torch.stack(b)

        # Assuming x is a list of node feature tensors and edge_index is shared
        # Create a list of Data objects
        data_list = [Data(x=feat, edge_index=edge_index) for feat in x]

        # Use Batch to process all Data objects at once
        batch = Batch.from_data_list(data_list)

        # Now pass the batched graph to your model
        batch_output = self.GCN3(batch.x, batch.edge_index)
        x1 = batch_output

        biases = self.biases[(output_size,)]
        x1 += biases
        x1 = x1.reshape(shape=(batch_size, self.num_nodes* output_size))
        return x1


class GraphGRU(nn.Module):
    def __init__(self,future, input_size, hidden_size, output_dim,history,num_nodes,r1,r2):
        super(GraphGRU, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim =input_size
        self.output_dim = output_dim
        self.gru_units = hidden_size
        self.r1 = r1
        self.r2 = r2
        self.input_window = history
        self.output_window = future
        self.device = torch.device('cuda')
        # add a cpu device for testing
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

        # -------------------构造模型-----------------------------
        self.GraphGRU_model = GraphGRUCell(self.gru_units, self.num_nodes, self.r1, self.r2, self.device, self.input_dim)
        self.GraphGRU_model1 = GraphGRUCell(self.gru_units, self.num_nodes, self.r1,self.r2, self.device, self.input_dim)
        self.fc1 = nn.Linear(self.gru_units*2, 120)
        #self.output_model = nn.Linear(self.gru_units*2, self.output_window * self.output_dim)
        self.output_model = nn.Linear(120, self.output_window * self.output_dim)
    def forward(self, x):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = x
        # labels = batch['y']

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        # assert batch_size == 200
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        state1 = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)

        for t in range(input_window):
              state = self.GraphGRU_model(inputs[t], state)
              state1 = self.GraphGRU_model1(inputs[input_window-t-1], state1)


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        state1 = state1.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        #output1 = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)

        state2 = torch.cat([state, state1], dim=2)
        
        state2=self.fc1(state2)
        state2 = state2.relu()
        output2=self.output_model(state2)
        state2 = state2.sigmoid()

        output2 = output2.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        output2 = output2.permute(0, 2, 1, 3)

        return output2

def eval(mymodel,test_loader,future):
    mae = MeanAbsoluteError().cuda()
    mape=MeanAbsolutePercentageError().cuda()
    mse=MeanSquaredError().cuda()
    net = mymodel.eval().cuda()
    real=[]
    prediction=[]
    history = []
    MAE=0
    MAPE=0
    MSE=0
    BAT_=0
    with torch.no_grad():
        for i,(batch_x, batch_y) in enumerate (test_loader):
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            outputs = net(batch_x)
            for u in range(future):

                # if u==2:
                #   real,prediction,history=save(batch_x,batch_y,outputs,real,prediction,history)
                MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                
                # MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MSE_d = mse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()

                # MAE+=MAE_d
                # MAPE+=MAPE_d
                # MSE+=MSE_d
                # BAT_+=1
                # import pdb;pdb.set_trace()
                # print("1 TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % ((u+1),MAE/BAT_, MAPE/BAT_,MSE/BAT_))
                print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % ((u+1),MAE_d, MAPE_d,MSE_d))
            #  if u==2:
            #     # import pdb; pdb.set_trace()
            #     with open('history.pkl', 'wb') as f:
            #         pickle.dump(history, f)
            #     with open('real.pkl', 'wb') as f:
            #         pickle.dump(real, f) 
            #     with open('prediction.pkl', 'wb') as f:
            #         pickle.dump(prediction, f)    





def eval_model():
    history,future=150,60
    output_size = 1
    # train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users(history,future)
    train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=1,p_end=3)
    print('shape of train_x:',train_x.shape,'shape of train_y:',train_y.shape,'shape of test_x:',test_x.shape,'shape of test_y:',test_y.shape)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    batch_size=test_x.shape[0]
    train_dataset=torch.utils.data.TensorDataset(train_x,train_y)
    test_dataset=torch.utils.data.TensorDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    ##################################################分界线##########################################
    # load model and test
    if not torch.cuda.is_available():
        mymodel = GraphGRU(future,feature_num,100,output_size,history)
    else:
        mymodel=GraphGRU(future,feature_num,100,output_size,history).cuda()
    mymodel.load_state_dict(torch.load(f'./data/graphgru_{70}.pkl')) 
    eval(mymodel,test_loader,future)



def main():
    test_flag = True
    voxel_size = int(128)
    num_nodes = 240
    history,future=90,60
    output_size = 1
    batch_size=64
    train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=1,p_end=4,voxel_size=voxel_size,num_nodes=num_nodes)
    print('shape of train_x:',train_x.shape,'shape of train_y:',train_y.shape,
          'shape of test_x:',test_x.shape,'shape of test_y:',test_y.shape,
          'shape of val_x:',val_x.shape,'shape of val_y:',val_y.shape)
    
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    
    train_dataset=torch.utils.data.TensorDataset(train_x,train_y)
    test_dataset=torch.utils.data.TensorDataset(test_x,test_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    # load graph edges
    voxel_size = int(256/2)
    edge_prefix = str(voxel_size)
    edge_path = f'./data/{edge_prefix}/graph_edges_integer_index.csv'
    # r1, r2 = getedge('newedge',900)
    r1, r2 = getedge(edge_path,4338)
    ##################################################分界线##########################################
    # write a cpu model for testing
        #  a.to(self.device)
    feature_num = train_x.shape[-1]
    assert feature_num == 7
    input_size = feature_num
    if not torch.cuda.is_available():
        mymodel = GraphGRU(future,input_size,100,output_size,history,num_nodes,r1,r2)   
    else:
        mymodel=GraphGRU(future,input_size,100,output_size,history,num_nodes,r1,r2).cuda()
    print(mymodel)
    num_epochs=64
    learning_rate=0.0003
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    output_windows=future
    lossa=[]
    for epochs in range(num_epochs):
        iter1 = 0
        iter2 = 0
        loss_total=0
        RMSET=0
        for i,(batch_x, batch_y) in tqdm(enumerate (train_loader)):
            #  import pdb;pdb.set_trace()
            #  print(f'{i}/{len(train_loader)}')
            if torch.cuda.is_available():
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
            else:
                batch_x=batch_x
                batch_y=batch_y
            #  import pdb;pdb.set_trace()   
            outputs = mymodel(batch_x)
            #  import pdb;pdb.set_trace()
            # clear the gradients
            optimizer.zero_grad()
            #loss
            loss = criterion(outputs,batch_y)
            loss_total=loss_total+loss.item()
            #backpropagation
            loss.backward()
            optimizer.step()
            iter1+=1
        loss_avg = loss_total/iter1
        losss=loss_avg
        lossa.append(losss)
        print("epoch:%d,  loss: %1.5f" % (epochs, loss_avg))
        # save model every 10 epochs and then reload it to continue training
        if epochs % 10 == 0:
            #save and reload
            torch.save(mymodel.state_dict(), f'./data/graphgru_{epochs}.pkl')
            #   mymodel.load_state_dict(torch.load(f'./data/graphgru_{epochs}.pkl')) 
            print('model saved')


    np.save('./data/graphgruloss.txt',lossa)
    print('loss saved')

    mae = MeanAbsoluteError().cuda()
    mape=MeanAbsolutePercentageError().cuda()
    mse=MeanSquaredError().cuda()
    net = mymodel.eval().cuda()
    real=[]
    prediction=[]
    history = []
    MAE=0
    MAPE=0
    MSE=0
    BAT_=0
    with torch.no_grad():
        if test_flag:
            for u in range(future):
                for i,(batch_x, batch_y) in enumerate (test_loader):
                    batch_x=batch_x.cuda()
                    batch_y=batch_y.cuda()
                    outputs = net(batch_x)
                    # if u==2:
                    #   real,prediction,history=save(batch_x,batch_y,outputs,real,prediction,history)
                    MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                    MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                    
                    # MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                    MSE_d = mse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()

                    MAE+=MAE_d
                    MAPE+=MAPE_d
                    MSE+=MSE_d
                    BAT_+=1
                print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % ((u+1),MAE/BAT_, MAPE/BAT_,MSE/BAT_))
                #  if u==2:
                #     # import pdb; pdb.set_trace()
                #     with open('history.pkl', 'wb') as f:
                #         pickle.dump(history, f)
                #     with open('real.pkl', 'wb') as f:
                #         pickle.dump(real, f) 
                #     with open('prediction.pkl', 'wb') as f:
                #         pickle.dump(prediction, f)    
if __name__ == '__main__':
    main()
    # eval_model()
