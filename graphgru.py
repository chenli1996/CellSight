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
from torch_geometric.data import Data
from tqdm import tqdm
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
    # print(len(df))
    # w
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
    x = df[data_type[2]].to_numpy()
    x_n = (x - min(x)) / (max(x) - min(x))

    # x_2= df[data_type[1]].to_numpy()

    x_list = []
    for feature in data_type:
        x_2 = df[feature].to_numpy()
        x_list.append((x_2 - min(x_2)) / (max(x_2) - min(x_2)))
        # x_list.append(x_2)


    #x_3= df[data_type[2]].to_numpy()
    # return x_n,x_2
    return x_n,x_list

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def get_train_data(data,history,future):
    data_x = []
    data_y = []
    for i in range(len(data)-history-future):
       data_x.append(data[i:i+history])
       data_y.append(data[i+history:i+history+future])
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    # data_y only get the third column feature of the last dim, from shape (batch size, 3, 240, 7)
    # to (batch size, 3, 240)
    data_y = data_y[:,:,:,2:3]#only occlusion feature

    size1 = int(len(data_x) * 0.8)
    size2 = int(len(data_x) * 1)
    train_x = data_x[:size1]
    train_y = data_y[:size1]
    test_x = data_x[size1:size2]
    test_y = data_y[size1:size2]
    val_x = data_x[size2:]
    val_y = data_y[size2:]
    # import pdb;pdb.set_trace()
    # train_x = np.expand_dims(train_x,axis=-1)
    # train_y = np.expand_dims(train_y,axis=-1)
    # test_x = np.expand_dims(test_x,axis=-1)
    # test_y = np.expand_dims(test_y,axis=-1)
    # val_x = np.expand_dims(val_x,axis=-1)
    # val_y = np.expand_dims(val_y,axis=-1)
    return train_x,train_y,test_x,test_y,val_x,val_y
history,future=150,60
voxel_size = int(256/2)
pcd_name = 'soldier'
# participant = 'P01_V1'
train_x,train_y,test_x,test_y,val_x,val_y = [],[],[],[],[],[]
for user_i in tqdm(range(1,17)):
    participant = 'P'+str(user_i).zfill(2)+'_V1'
    # generate graph voxel grid features
    prefix = f'{pcd_name}_VS{voxel_size}'
    node_feature_path = f'./data/{prefix}/{participant}node_feature.csv'
    column_name = ['occupancy_feature','in_FoV_feature','occlusion_feature','coordinate_x','coordinate_y','coordinate_z','distance']
    # column_name ['occlusion_feature']
    a1,a2=getdata_normalize(node_feature_path,column_name)
    # x=np.array(list(zip(a1)))
    # x=np.array(list(zip(a2)))
    x=np.array(a2)
    # x=x.reshape(1440,301,1)
    feature_num = len(column_name)
    # feature_num = 1
    print('feature_num:',feature_num)
    x=x.reshape(feature_num,150,240)
    x=x.transpose(1,2,0)
    train_x1,train_y1,test_x1,test_y1,val_x1,val_y1=get_train_data(x,history,future)
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




train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)


batch_size=32
train_dataset=torch.utils.data.TensorDataset(train_x,train_y)
test_dataset=torch.utils.data.TensorDataset(test_x,test_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#######################################################
class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.DoubleTensor(self._num_gru_units + feature_num, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        num_nodes=240
        batch_size = hidden_state.shape[0]
        # assert batch_size == 200
        inputs = inputs.reshape((batch_size, num_nodes, feature_num))
        # inputs (batch_size, num_nodes, feature_num)
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_gru_units + feature_num))
        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
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
    def __init__(self, num_units, num_nodes, device, input_dim=1):
        super(GraphGRUCell, self).__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.device = device
        self.act = torch.tanh
        self.init_params()
        # 这里提前构建好边集
        voxel_size = int(256/2)
        edge_prefix = str(voxel_size)
        edge_path = f'./data/{edge_prefix}/graph_edges_integer_index.csv'
        # r1, r2 = getedge('newedge',900)
        r1, r2 = getedge(edge_path,4338)
        self.r1 = r1
        self.r2 = r2
        self.GRU1 = GRULinear(100, 200)
        self.GRU2 = GRULinear(100, 100)
        # self.GCN3 = GATConv(101, 100)
        self.GCN3 = GATConv(100+feature_num, 100)
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
        for i in x:
          x111=Data(x=i,edge_index=edge_index)
          xx=self.GCN3(x111.x,x111.edge_index)
          b.append(xx)
        x1=torch.stack(b)
        biases = self.biases[(output_size,)]
        x1 += biases
        x1 = x1.reshape(shape=(batch_size, self.num_nodes* output_size))
        return x1


class GraphGRU(nn.Module):
    def __init__(self,future, input_size, hidden_size, output_dim,inputwindow):
        super(GraphGRU, self).__init__()
        self.num_nodes = 240
        self.input_dim =input_size
        self.output_dim = output_dim
        self.gru_units = hidden_size

        self.input_window = inputwindow
        self.output_window = future
        self.device = torch.device('cuda')
        # add a cpu device for testing
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

        # -------------------构造模型-----------------------------
        self.GraphGRU_model = GraphGRUCell(self.gru_units, self.num_nodes, self.device, self.input_dim)
        self.GraphGRU_model1 = GraphGRUCell(self.gru_units, self.num_nodes, self.device, self.input_dim)
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



##################################################分界线##########################################
# write a cpu model for testing
    #  a.to(self.device)
input_size = feature_num
out_size = 1
if not torch.cuda.is_available():
    mymodel = GraphGRU(future,feature_num,100,out_size,history)
else:
    mymodel=GraphGRU(future,feature_num,100,out_size,history).cuda()
print(mymodel)
num_epochs=100
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
  for i,(batch_x, batch_y) in enumerate (train_loader):
     
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
np.save('lossa',lossa)
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
 for u in range(future):
     for i,(batch_x, batch_y) in enumerate (test_loader):
        batch_x=batch_x.cuda()
        batch_y=batch_y.cuda()
        outputs = net(batch_x)
        if u==2:
          real,prediction,history=save(batch_x,batch_y,outputs,real,prediction,history)
        MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        
        # MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
        MSE_d = mse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()

        MAE+=MAE_d
        MAPE+=MAPE_d
        MSE+=MSE_d
        BAT_+=1
     print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % (30*(u+1),MAE/BAT_, MAPE/BAT_,MSE/BAT_))
     if u==2:
        # import pdb; pdb.set_trace()
        with open('history.pkl', 'wb') as f:
            pickle.dump(history, f)
        with open('real.pkl', 'wb') as f:
            pickle.dump(real, f) 
        with open('prediction.pkl', 'wb') as f:
            pickle.dump(prediction, f)    

