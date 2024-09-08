#!/bin/env python

from audioop import rms
from cgi import test
from math import e
from operator import le
from re import T
from turtle import forward
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torch_geometric.nn import GATConv,GATv2Conv,TransformerConv
from torch_geometric.data import Data,Batch
from tqdm import tqdm
from time import time
import os
from utils_graphgru import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.preprocessing import MinMaxScaler

# torch.autograd.set_detect_anomaly(True)


# torch.set_default_tensor_type(torch.DoubleTensor)
# set to float32
# torch.set_default_dtype(torch.float32)
# torch.set_default_device()
# torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)
# torch.set_default_device('cpu')  # or 'cuda' if you're using a GPU
# torch.set

#######################################################
class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int,num_nodes: int, feature_num: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.feature_num = feature_num
        self.weights = nn.Parameter(
            # torch.DoubleTensor(self._num_gru_units + self.feature_num, self._output_dim)
            torch.FloatTensor(self._num_gru_units + self.feature_num, self._output_dim)
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
    def __init__(self, num_units, num_nodes, r1,r2, batch_size,device, input_dim=1):
        super(GraphGRUCell, self).__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.device = device
        self.act = torch.tanh
        self.init_params()
        self.r1 = r1
        self.r2 = r2
        self.GRU1 = GRULinear(self.num_units, 2 * self.num_units, self.num_nodes,self.input_dim)
        self.GRU2 = GRULinear(self.num_units, self.num_units, self.num_nodes,self.input_dim)
        # self.GCN3 = GATConv(101, 100)
        # Precompute edge_index_expanded
        self.edge_index_expanded = self.precompute_edge_index(self.batch_size)

        self.head = 1
        self.multiGAT = False
        self.dropout = 0.2
        self.OriginalGAT = True
        # self.GCN3 = GATConv(self.num_units+self.input_dim, self.num_units)
        if self.OriginalGAT:
            self.GCN3 = GATConv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            self.GCN4 = GATConv(self.num_units,self.num_units,concat=False)
        else:
            # self.GAT3 = GATv2Conv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            # self.GAT4 = GATv2Conv(self.num_units*self.head,self.num_units,concat=False)

            self.GAT3 = TransformerConv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            self.GAT4 = TransformerConv(self.num_units,self.num_units,concat=False)


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

    def precompute_edge_index(self, batch_size=None):
        edge_index = torch.tensor(np.stack((np.array(self.r1),np.array(self.r2))), dtype=torch.long).to(self.device)
        # Ensure edge_index is on the GPU
        edge_index = edge_index.to(self.device)
        
        # Replicate edge_index for each graph in the batch
        edge_index_expanded = edge_index.repeat(1, batch_size)
        
        # Create edge_index_offsets directly on the GPU
        edge_index_offsets = torch.arange(batch_size, device=self.device).repeat_interleave(edge_index.size(1)) * self.num_nodes
        
        # Add the offsets to edge_index_expanded
        edge_index_expanded += edge_index_offsets
        
        return edge_index_expanded        

    def forward(self, inputs, state):
        # inputs (batch_size, num_nodes * input_dim)
        # state (batch_size, num_nodes * gru_units) or (batch_size, num_nodes* num_units)
        batch_size = state.shape[0]
        # import pdb;pdb.set_trace()
        # update state using graph neighbors
        state=self._gc3(state,inputs, self.num_units) # (batch_size, self.num_nodes * self.gru_units)
        output_size = 2 * self.num_units
        value = torch.sigmoid(
            self.GRU1(inputs, state))  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1) # (batch_size, self.num_nodes, self.gru_units)
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
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1)) # (batch, self.num_nodes, self.input_dim)
        inputs_and_state = torch.cat([state, inputs], dim=2)
        input_size = inputs_and_state.shape[2]
        x = inputs_and_state.to(self.device)
        # edge_index = torch.tensor([self.r1, self.r2], dtype=torch.long).to(self.device)
        # import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()
        # b=[]
        # for i in x:
        #   x111=Data(x=i,edge_index=edge_index)
        #   xx=self.GCN3(x111.x,x111.edge_index)
        #   b.append(xx)
        # x1=torch.stack(b)

        # Assuming x is a list of node feature tensors and edge_index is shared
        # Create a list of Data objects
        # edge_index = torch.tensor([self.r1, self.r2], dtype=torch.long).to(self.device)
        # data_list = [Data(x=feat, edge_index=edge_index) for feat in x]

        # Use Batch to process all Data objects at once
        # batch1 = Batch.from_data_list(data_list)
        # Now pass the batched graph to your model
        # batch_output1 = self.GCN3(batch1.x, batch1.edge_index)

        # Flatten the input tensor and create a large batch of node features
        batch_size, num_nodes, num_features = x.size()
        x_flat = x.view(-1, num_features)  # Shape: (batch_size * num_nodes, num_features)

        # # Replicate edge_index for each graph in the batch
        # edge_index_expanded = edge_index.repeat(1, batch_size)
        # edge_index_offsets = torch.arange(batch_size).repeat_interleave(edge_index.size(1)) * num_nodes
        # edge_index_expanded += edge_index_offsets.to(self.device)

        # Create a single Data object and then batch it
        if batch_size == self.batch_size:
            edge_index = self.edge_index_expanded
        else:
            # last batch may have fewer samples
            edge_index = self.precompute_edge_index(batch_size)
        data = Data(x=x_flat, edge_index=edge_index)
        batch = Batch.from_data_list([data])
        # import pdb;pdb.set_trace()
        # Pass the batched graph to the model
        if self.OriginalGAT:
            if self.multiGAT:
                x = self.GCN3(batch.x, batch.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
                x = self.GCN4(x, batch.edge_index)
                x = F.relu(x)
            else:
                x = self.GCN3(batch.x, batch.edge_index)
        else:
            if self.multiGAT:
                x = self.GAT3(batch.x, batch.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
                x = self.GAT4(x, batch.edge_index)
                x = F.relu(x)
            else:
                x = self.GAT3(batch.x, batch.edge_index)

        # biases = self.biases[(output_size,)]
        # x += biases
        x = x.reshape(shape=(batch_size, self.num_nodes* output_size))
        # import pdb;pdb.set_trace()
        return x


class GraphGRU(nn.Module):
    def __init__(self,future, input_size, hidden_size, output_dim,history,num_nodes,r1,r2,batch_size=128):
        super(GraphGRU, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim =input_size
        self.output_dim = output_dim
        self.gru_units = hidden_size
        self.r1 = r1
        self.r2 = r2
        self.batch_size = batch_size
        self.input_window = history
        self.output_window = future
        self.device = torch.device('cuda')
        # add a cpu device for testing
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        self.GraphGRU_model = GraphGRUCell(self.gru_units, self.num_nodes, self.r1, self.r2, self.batch_size, self.device, self.input_dim)
        self.GraphGRU_model1 = GraphGRUCell(self.gru_units, self.num_nodes, self.r1,self.r2, self.batch_size, self.device, self.input_dim)
        self.GraphGRU_model_o = GraphGRUCell(self.gru_units, self.num_nodes, self.r1,self.r2, self.batch_size, self.device, self.output_dim+1)
        self.GraphGRU_model_future = GraphGRUCell(self.gru_units, self.num_nodes, self.r1,self.r2, self.batch_size, self.device, self.input_dim)
        self.fc1 = nn.Linear(self.gru_units*2, self.gru_units)
        self.output_window = 1
        self.output_model = nn.Linear(self.gru_units, self.output_window * self.output_dim)
        # self.combine_O_L_F = nn.Linear(self.output_dim+1, self.output_dim)
        # MLP layer for combine_O_L_F
        self.combine_O_L_F = nn.Sequential(
                    nn.Linear(self.gru_units, self.output_dim*32, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.output_dim*32, self.output_dim*32, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.output_dim*32, self.output_dim, bias=True),
                    nn.ReLU()
                )

        self.edge_index_expanded = self.precompute_edge_index(self.batch_size)

        self.head = 1
        self.multiGAT = False
        self.dropout = 0.2
        self.OriginalGAT = False
        self.num_units = hidden_size
        self.bi_rnn = True
        # self.GCN3 = GATConv(self.num_units+self.input_dim, self.num_units)
        if self.OriginalGAT:
            self.GCN3_1 = GATConv(self.num_units+output_dim+1, self.num_units,heads=self.head,concat=False)
            self.GCN4_1 = GATConv(self.num_units,self.num_units,concat=False)
        else:
            # self.GAT3 = GATv2Conv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            # self.GAT4 = GATv2Conv(self.num_units*self.head,self.num_units,concat=False)

            self.GAT3_1 = TransformerConv(self.num_units+output_dim+1, self.num_units,heads=self.head,concat=False)
            self.GAT4_1 = TransformerConv(self.num_units,self.num_units,concat=False)
        self.afterG = nn.Linear(self.num_units, self.output_dim)
        self.mylinear = nn.Linear(self.output_dim, self.output_dim)
        self.batch_norm = nn.BatchNorm1d(self.input_dim)

    def precompute_edge_index(self, batch_size=None):
        edge_index = torch.tensor(np.stack((np.array(self.r1),np.array(self.r2))), dtype=torch.long).to(self.device)
        # Ensure edge_index is on the GPU
        edge_index = edge_index.to(self.device)
        
        # Replicate edge_index for each graph in the batch
        edge_index_expanded = edge_index.repeat(1, batch_size)
        
        # Create edge_index_offsets directly on the GPU
        edge_index_offsets = torch.arange(batch_size, device=self.device).repeat_interleave(edge_index.size(1)) * self.num_nodes
        
        # Add the offsets to edge_index_expanded
        edge_index_expanded += edge_index_offsets
        
        return edge_index_expanded   


        # self.output_model = nn.Linear(self.gru_units, self.output_window * self.output_dim)
        # only predict one future frame
        # self.fc2 = nn.Linear(self.gru_units, self.output_dim) #used to get state for F^ and L^
        # self.fc2s = nn.ModuleList([nn.Linear(self.gru_units, self.output_dim) for i in range(self.output_window)])
        # self.fc2_F_Ls = nn.ModuleList([nn.Linear(self.output_dim+1, self.output_dim) for i in range(self.output_window)])
        # self.fc2_F_L = nn.Linear(self.input_dim-1, self.output_dim) #used to get F^ and L^ using O^x
        # self.output_model = nn.Linear(self.gru_units, self.output_dim)
        # I want to have  output_models with the number of self.output_window
        # self.output_models = nn.ModuleList([nn.Linear(self.gru_units, self.output_dim) for i in range(self.output_window)])

        

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

# ______________________________________________________________________________________________________________________
        # Reshape x to [input_window * batch_size * num_nodes, input_dim]
        # x_reshaped = x.permute(1, 2, 0, 3).contiguous().view(-1, input_dim)
        # # Apply BatchNorm
        # x_normalized = self.batch_norm(x_reshaped)

        # # Reshape back to the original shape if necessary
        # x_normalized = x_normalized.view(input_window, batch_size, num_nodes, input_dim).permute(2, 1, 0, 3)
        # inputs = x_normalized.contiguous()

# ______________________________________________________________________________________________________________________

        
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device) # (input_window, batch_size, num_nodes * input_dim)
        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device) # (batch_size, self.num_nodes * self.gru_units)
        state1 = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)

        for t in range(input_window):
              state = self.GraphGRU_model(inputs[t], state) # (batch_size, self.num_nodes * self.gru_units)
              state1 = self.GraphGRU_model1(inputs[input_window-t-1], state1) # (batch_size, self.num_nodes * self.gru_units)


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        state1 = state1.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        #output1 = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)

        if self.bi_rnn:
        #bi-rnn
            state2 = torch.cat([state, state1], dim=2) # (batch_size, self.num_nodes, self.gru_units*2)
            state2=self.fc1(state2) # (batch_size, self.num_nodes, self.gru_units)
        #no bi-rnn
        else:
            state2 = state


        state2 = state2.relu()
        output2=self.output_model(state2) # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        # state2 = state2.sigmoid()

        # self.output_window = 1
        output2 = output2.view(batch_size, self.num_nodes, self.output_window, self.output_dim) # (batch_size, self.num_nodes, self.output_window, self.output_dim)
        output2 = output2.permute(0, 2, 1, 3) # (batch_size, self.output_window, self.num_nodes, self.output_dim)

        # 
        return output2.sigmoid()
    
    def forward_output1_o(self,x,y):
        # batch normalization
        inputs = x
        # labels = batch['y']

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        # assert batch_size == 200
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)

# ______________________________________________________________________________________________________________________
        # # Reshape x to [input_window * batch_size * num_nodes, input_dim]
        # x_reshaped = x.permute(1, 2, 0, 3).contiguous().view(-1, input_dim)
        # # Apply BatchNorm
        # x_normalized = self.batch_norm(x_reshaped)

        # # Reshape back to the original shape if necessary
        # x_normalized = x_normalized.view(input_window, batch_size, num_nodes, input_dim).permute(2, 1, 0, 3)
        # inputs = x_normalized.contiguous()

# ______________________________________________________________________________________________________________________

        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device) # (input_window, batch_size, num_nodes * input_dim)


        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device) # (batch_size, self.num_nodes * self.gru_units)
        state1 = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)

        for t in range(input_window):
              state = self.GraphGRU_model(inputs[t], state) # (batch_size, self.num_nodes * self.gru_units)
              state1 = self.GraphGRU_model1(inputs[input_window-t-1], state1) # (batch_size, self.num_nodes * self.gru_units)


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        state1 = state1.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        #output1 = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)

        state2 = torch.cat([state, state1], dim=2) # (batch_size, self.num_nodes, self.gru_units*2)
        # import pdb;pdb.set_trace()
        state2=self.fc1(state2) # (batch_size, self.num_nodes, self.gru_units)
        state2 = state2.relu()
        output2=self.output_model(state2) # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        # state2 = state2.sigmoid()
        # self.output_window = 1
        output2 = output2.view(batch_size, self.num_nodes, self.output_window, self.output_dim) # (batch_size, self.num_nodes, self.output_window, self.output_dim)
        output2 = output2.permute(0, 2, 1, 3) # (batch_size, self.output_window, self.num_nodes, self.output_dim)
        output2 = output2.sigmoid() # (batch_size, self.output_window, self.num_nodes, self.output_dim)
        y_occupancy = y[:,-1,:,1].unsqueeze(1).unsqueeze(3) # (batch_size, 1, num_nodes, 1) !!!!!!!!!!!!!!!!!!!change to -30 or -1
        # import pdb;pdb.set_trace()

        
        # # import pdb;pdb.set_trace()
        O_L = torch.cat([y_occupancy,output2],dim=3) # (batch_size, 1, num_nodes, output_dim+1)
        # O_L = y_occupancy + output2
        # output = self.mylinear(O_L)
        # output = y_occupancy.sigmoid()



        state2 = state2.view(batch_size, self.num_nodes, self.gru_units)
        # output = self.GraphGRU_model_o(O_L, state2)
        # output = output.view(batch_size, self.num_nodes, self.gru_units)
        # output = self.combine_O_L_F(output)
        # output = output.sigmoid()
        # output.unsqueeze(1)

        # import pdb;pdb.set_trace()
        O_L = O_L.squeeze(1)
        x = torch.cat([state2,O_L],dim=2)
        # import pdb;pdb.set_trace()
        batch_size, num_nodes, num_features = x.size()
        x_flat = x.view(-1, num_features)  # Shape: (batch_size * num_nodes, num_features)

        # # Replicate edge_index for each graph in the batch
        # edge_index_expanded = edge_index.repeat(1, batch_size)
        # edge_index_offsets = torch.arange(batch_size).repeat_interleave(edge_index.size(1)) * num_nodes
        # edge_index_expanded += edge_index_offsets.to(self.device)

        # Create a single Data object and then batch it
        if batch_size == self.batch_size:
            edge_index = self.edge_index_expanded
        else:
            # last batch may have fewer samples
            edge_index = self.precompute_edge_index(batch_size)
        data = Data(x=x_flat, edge_index=edge_index)
        batch = Batch.from_data_list([data])
        # import pdb;pdb.set_trace()
        # Pass the batched graph to the model
        if self.OriginalGAT:
            if self.multiGAT:
                x = self.GCN3_1(batch.x, batch.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
                x = self.GCN4_1(x, batch.edge_index)
                x = F.relu(x)
            else:
                x = self.GCN3_1(batch.x, batch.edge_index)
        else:
            if self.multiGAT:
                x = self.GAT3_1(batch.x, batch.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
                x = self.GAT4_1(x, batch.edge_index)
                x = F.relu(x)
            else:
                x = self.GAT3_1(batch.x, batch.edge_index)

        # biases = self.biases[(output_size,)]
        # x += biases
        # x = x.reshape(shape=(batch_size, self.num_nodes* output_size))
        # import pdb;pdb.set_trace()
        x = x.reshape(shape=(batch_size, self.num_nodes, self.num_units))
        x = self.afterG(x)
        x = x.relu()
        x = x.view(batch_size, self.num_nodes, self.output_dim)
        output = x.sigmoid()
        output = output.unsqueeze(1)

        
        return output
        # 

    def forward_object(self, x, y):
        # y[:,:,:,1:3] = -100
        # y[:,:,:,6] = -100
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
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device) # (input_window, batch_size, num_nodes * input_dim)
        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device) # (batch_size, self.num_nodes * self.gru_units)
        state1 = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)

        for t in range(input_window):
              state = self.GraphGRU_model(inputs[t], state) # (batch_size, self.num_nodes * self.gru_units)
              state1 = self.GraphGRU_model1(inputs[input_window-t-1], state1) # (batch_size, self.num_nodes * self.gru_units)
            #   here we may need activation function after each state


        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        state1 = state1.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        #output1 = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)

        # state2 = torch.cat([state, state1], dim=2) # (batch_size, self.num_nodes, self.gru_units*2)
        state_combined = torch.cat([state, state1], dim=2) # (batch_size, self.num_nodes, self.gru_units*2)
        state_transformed = self.fc1(state_combined)
        state_transformed = state_transformed.relu()
        # import pdb;pdb.set_trace()
        # state2=self.fc1(state2) # (batch_size, self.num_nodes, self.gru_units)
        # state2 = state2.relu()
        # Output initialization
        outputs = []
        new_state = state_transformed
        F_L_state = self.fc2s[0](new_state) # (batch_size, self.num_nodes, self.output_dim)
        F_L_state = F_L_state.view(batch_size, self.num_nodes, self.output_dim).sigmoid() # (batch_size, self.num_nodes, self.output_dim)
        # import pdb;pdb.set_trace()
        # get O,F^,L^:
        # import pdb;pdb.set_trace()
        F_L_hat_0_cat_o = torch.cat([y[:,0,:,0].unsqueeze(2), F_L_state], dim=2)
        
        # F_L_hat_0_cat_o.unsqueeze(2)
        F_L_hat_0 = self.fc2_F_Ls[0](F_L_hat_0_cat_o)
        F_L_hat_0 = F_L_hat_0.relu()
        F_L_hat_0 = F_L_hat_0.view(batch_size, self.num_nodes, self.output_dim).sigmoid() # (batch_size, self.num_nodes, self.output_dim)
        # current_input = F_L_hat_0
        
        F_L_hat = F_L_hat_0
        outputs.append(F_L_hat_0)
        # Generate predictions using the output of the model as new input
        for u in range(self.output_window-1):
            # Assume the output needs to be processed similarly through the GRUs
            y[:, u, :, 3-self.output_dim:3] = F_L_hat
            # import pdb;pdb.set_trace()
            current_input = y[:, u, :, :].view(batch_size, num_nodes * self.input_dim) # (O, F_hat, L_hat) (batch_size, num_nodes * input_dim)
            new_state = self.GraphGRU_model_future(current_input,new_state)
            new_state = new_state.view(batch_size, self.num_nodes, self.gru_units)
            F_L_state = self.fc2s[u+1](new_state) # (batch_size, self.num_nodes, self.output_dim)
            F_L_state = F_L_state.view(batch_size, self.num_nodes, self.output_dim).sigmoid() # (batch_size, self.num_nodes, self.output_dim)
            # get O,F^,L^:
            # import pdb;pdb.set_trace()
            # y[:, u+1, :, 3-self.output_dim:3] = F_L_state
            # F_L_hat_cat_o = y[:, u+1, :, 3-self.output_dim:3]
            F_L_hat_cat_o = torch.cat([y[:,u+1,:,0].unsqueeze(2), F_L_state], dim=2)
            F_L_hat = self.fc2_F_Ls[u+1](F_L_hat_cat_o)
            F_L_hat = F_L_hat.relu()
            F_L_hat = F_L_hat.view(batch_size, self.num_nodes, self.output_dim).sigmoid() # (batch_size, self.num_nodes, self.output_dim)
            outputs.append(F_L_hat)

        # Stack outputs to match expected dimensions
        outputs = torch.stack(outputs, dim=1)  # (batch_size, output_window, num_nodes, output_dim)
        # import pdb;pdb.set_trace()






        # output2=self.output_model(state2) # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        # state2 = state2.sigmoid()

        # self.output_window = 1
        # output2 = output2.view(batch_size, self.num_nodes, self.output_window, self.output_dim) # (batch_size, self.num_nodes, self.output_window, self.output_dim)
        # output2 = output2.permute(0, 2, 1, 3) # (batch_size, self.output_window, self.num_nodes, self.output_dim)

        # 
        return outputs
    

         

def eval_model(mymodel,test_loader,model_prefix,history=90,future=60):
    mae = MeanAbsoluteError().cuda()
    mape=MeanAbsolutePercentageError().cuda()
    mse=MeanSquaredError().cuda()
    net = mymodel.eval().cuda()
    mse_list = []
    mae_list = []
    mape_list = []
    with torch.no_grad():
        for i,(batch_x, batch_y) in enumerate (test_loader):
            assert i == 0 # batch size is equal to the test set size
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()


            outputs = net(batch_x)
            outputs,batch_y = mask_outputs_batch_y(outputs, batch_y)
            # batch_y = batch_y[:,:,:,2:3]
            for u in range(future):
                MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MSE_d = mse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()
                print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % ((u+1),MAE_d, MAPE_d,MSE_d))
                # import pdb;pdb.set_trace()
                if u==149:
                    for sample in range(0,batch_x.shape[0],100):
                        print('sample:',sample)
                        # print('output:',outputs[sample,u,:].view(30,8))
                        # print('label:',batch_y[sample,u,:].view(30,8))
                        print('output:',outputs[sample,u,134])
                        print('label:',batch_y[sample,u,134])
                        # import pdb;pdb.set_trace()
                mse_list.append(MSE_d.item())
                mae_list.append(MAE_d.item())
                mape_list.append(MAPE_d.item())
        print('MSE:',mse_list)
        print('MAE:',mae_list)
        # print('MAPE:',mape_list)
        # plot mse and mae
        plt.figure()
        plt.plot(mse_list)
        plt.plot(mae_list)
        # plt.plot(mape_list)
        plt.legend(['MSE', 'MAE'])
        plt.xlabel('Prediction Horizon/frame')
        plt.ylabel('Loss')
        plt.savefig(f'./data/fig/per_graphgru_{model_prefix}_testingloss{history}_{future}.png') 

def eval_model_sample(mymodel,test_loader,model_prefix,output_size,history=90,future=60, target_output=1,predict_index_end=3):
    mae = MeanAbsoluteError().cuda()
    mape=MeanAbsolutePercentageError().cuda()
    mse=MeanSquaredError().cuda()
    rmse = MeanSquaredError(squared=False).cuda()
    net = mymodel.eval().cuda()
    mse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []
    MAE = {}
    MAPE = {}
    MSE = {}
    RMSE = {}
    # criterion = torch.nn.MSELoss()    

    for i in range(future):
        MAE[i] = 0
        MAPE[i] = 0
        MSE[i] = 0
        RMSE[i] = 0

    with torch.no_grad():
        # import pdb;pdb.set_trace()

        for i,(batch_x, batch_y) in enumerate (test_loader):
            # assert i == 0 # batch size is equal to the test set size

            # import pdb;pdb.set_trace()
            # only predict last frame------
            batch_y=batch_y[:,-target_output,:,:]
            # keep batch_y as (batch_size, 1, self.num_nodes, self.output_dim)
            batch_y = batch_y.unsqueeze(1) 
            # ----------------------------- 

            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            outputs = net(batch_x) # (batch_size, self.output_window, self.num_nodes, self.output_dim)
            # -------------
            if predict_index_end==3:
                outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size,predict_index_end)
            else:
                batch_y = batch_y[:,:,:,predict_index_end-output_size:predict_index_end] # (batch_size, 1, self.num_nodes, output_dim)
            # ----------------
            # if i==0:
            # u = future-1
            # for index in range(0,outputs.size(0),1):
            #     if index+i*outputs.size(0)==1682:
            #         print('Graph',outputs[index, :, :, :].view(30,8))
            #         print('GT',batch_y[index, :, :, :].view(30,8))
            #     # import pdb;pdb.set_trace()
            #     MSE_temp = mse(batch_y[index, :, :, :].contiguous(), outputs[index, :, :, :].contiguous()).cpu().detach().numpy()
            #     MAE_temp = mae(batch_y[index, :,:,:],outputs[index,:,:,:]).cpu().detach().numpy()
            #     if abs(MSE_temp-0.072)<0.05 and MAE_temp<0.10:
            #         print(f'MSE:{MSE_temp},MAE:{MAE_temp}',f'index:{index+i*outputs.size(0)}')

            for u in range(outputs.shape[1]):
                MAE_d=mae(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MAPE_d=mape(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                # MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MSE_d = mse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()
                RMSE_d = rmse(outputs[:, u, :, :].contiguous(), batch_y[:, u, :, :].contiguous()).cpu().detach().numpy()

                MAE[u] += MAE_d
                MAPE[u] += MAPE_d
                MSE[u] += MSE_d
                RMSE[u] += RMSE_d
        for u in range(outputs.shape[1]):
            MAE_u = MAE[u]/(i+1)
            MAPE_u = MAPE[u]/(i+1)
            MSE_u = MSE[u]/(i+1)
            RMSE_u = RMSE[u]/(i+1)
            print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f, RMSE: %1.5f" % ((u+1),MAE_u, MAPE_u,MSE_u,RMSE_u))
        # import pdb;pdb.set_trace()
        # if u==149:
        #     for sample in range(0,batch_x.shape[0],100):
        #         print('sample:',sample)
        #         # print('output:',outputs[sample,u,:].view(30,8))
        #         # print('label:',batch_y[sample,u,:].view(30,8))
        #         print('output:',outputs[sample,u,134])
        #         print('label:',batch_y[sample,u,134])
        #         # import pdb;pdb.set_trace()
            mse_list.append(MSE_u)
            mae_list.append(MAE_u)
            mape_list.append(MAPE_u)
            rmse_list.append(RMSE_u)

        print('MSE:',mse_list)
        print('MAE:',mae_list)
        print('RMSE:',rmse_list)
        # print('MAPE:',mape_list)
        # plot mse and mae
        plt.figure()
        if len(mse_list) == 1:
            plt.scatter(future, mse_list[0])
        else:
            plt.plot(mse_list)

        if len(mae_list) == 1:
            plt.scatter(future, mae_list[0])
        else:
            plt.plot(mae_list)
        # plt.plot(mape_list)
        plt.legend(['MSE', 'MAE'])
        plt.xlabel('Prediction Horizon/frame')
        plt.ylabel('Loss')
        plt.savefig(f'./data/fig/p90_vs128_graphgru_{model_prefix}_testingloss{history}_{future}.png') 


def eval_model_sample_num(mymodel,test_loader,test_loader_nn,model_prefix,output_size,history=90,future=60):


    mae = MeanAbsoluteError().cuda()
    mape=MeanAbsolutePercentageError().cuda()
    mse=MeanSquaredError().cuda()
    net = mymodel.eval().cuda()
    mse_list = []
    mae_list = []
    mape_list = []
    MAE = {}
    MAPE = {}
    MSE = {}


    for i in range(future):
        MAE[i] = 0
        MAPE[i] = 0
        MSE[i] = 0

    with torch.no_grad():
        for i,(batch_x, batch_y),(batch_x_nn,batch_y_nn) in enumerate(zip(test_loader,test_loader_nn)):
            print(i)
            # assert i == 0 # batch size is equal to the test set size

            # only predict last frame------
            # batch_y=batch_y[:,-target_output,:,:]
            # keep batch_y as (batch_size, 1, self.num_nodes, self.output_dim)
            # batch_y = batch_y.unsqueeze(1) 
            # ----------------------------- 
            


            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            batch_x_nn = batch_x_nn.cuda()
            outputs = net(batch_x)
            batch_y_occupancy = batch_y_nn[:,:,:,0].clone().unsqueeze(3)

            outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size)
            # for sample in range(0,batch_x.shape[0],100):
            # batch_y = batch_y[:,:,:,2:3]
            # import pdb;pdb.set_trace() 
            for u in range(outputs.shape[1]):
                import pdb;pdb.set_trace()
                MAE_d=mae(outputs[:,u,:,:]*batch_y_occupancy[:,u,:,:],batch_y[:,u,:,:]*batch_y_occupancy[:,u,:,:]).cpu().detach().numpy()
                MAPE_d=mape(outputs[:,u,:,:]*batch_y_occupancy[:,u,:,:],batch_y[:,u,:,:]*batch_y_occupancy[:,u,:,:]).cpu().detach().numpy()
                # MSE_d=mse(outputs[:,u,:,:],batch_y[:,u,:,:]).cpu().detach().numpy()
                MSE_d = mse(outputs[:, u, :, :].contiguous()*batch_y_occupancy[:,u,:,:], batch_y[:, u, :, :].contiguous()*batch_y_occupancy[:,u,:,:]).cpu().detach().numpy()

                MAE[u] += MAE_d
                MAPE[u] += MAPE_d
                MSE[u] += MSE_d
        for u in range(outputs.shape[1]):
            MAE_u = MAE[u]/(i+1)
            MAPE_u = MAPE[u]/(i+1)
            MSE_u = MSE[u]/(i+1)
            print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f" % ((u+1),MAE_u, MAPE_u,MSE_u))
        # import pdb;pdb.set_trace()
        # if u==149:
        #     for sample in range(0,batch_x.shape[0],100):
        #         print('sample:',sample)
        #         # print('output:',outputs[sample,u,:].view(30,8))
        #         # print('label:',batch_y[sample,u,:].view(30,8))
        #         print('output:',outputs[sample,u,134])
        #         print('label:',batch_y[sample,u,134])
        #         # import pdb;pdb.set_trace()
            mse_list.append(MSE_u)
            mae_list.append(MAE_u)
            mape_list.append(MAPE_u)
        print('MSE:',mse_list)
        print('MAE:',mae_list)
        # print('MAPE:',mape_list)
        # plot mse and mae
        plt.figure()
        if len(mse_list) == 1:
            plt.scatter(future, mse_list[0])
        else:
            plt.plot(mse_list)

        if len(mae_list) == 1:
            plt.scatter(future, mae_list[0])
        else:
            plt.plot(mae_list)
        # plt.plot(mape_list)
        plt.legend(['MSE', 'MAE'])
        plt.xlabel('Prediction Horizon/frame')
        plt.ylabel('Loss')
        plt.savefig(f'./data/fig/per2num_p150_vs128_graphgru_{model_prefix}_testingloss{history}_{future}.png') 


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-2):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply sigmoid to the predicted outputs to get probabilities
        preds = torch.sigmoid(preds)
        targets = torch.sigmoid(targets)
        
        # Flatten the tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        # Calculate Soft Dice coefficient
        soft_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Soft Dice Loss
        return 1.0 - soft_dice
    
def main(future=10):
    with_train = True
    continue_train_early_stop_val = False
    last_val_loss = 0.210087
    object_driven = False
    voxel_size = int(128)
    if voxel_size == 128:
        num_nodes = 240
    elif voxel_size == 64:
        num_nodes = 1728
    else:
        num_nodes = None
    history = 90
    future=future
    # history,future=3,10
    target_output = 1
    p_start = 1
    p_end = 28
    # p_end = 4
    output_size = 1
    predict_index_end=2
    num_epochs=50
    # batch_size = 16
    batch_size=32 #multi_out
    # batch_size=32 #G1 90
    # batch_size=64 # 256 model
    # batch_size=64*2 #150 64GB
    # batch_size=25 #G2 T h2
    # batch_size=32 #T1 h1 fulledge
    hidden_dim = 128

    # clip = 600
    # model_prefix = f'out1_pred_end2_90_10f_p1_skip1_num_G2_h1_fulledge_loss_part_{hidden_dim}_{voxel_size}'
    # model_prefix = f'out1_pred_end2_90_10f_p1_skip1_num_G2_h1_fulledge_100_128'
    # model_prefix = f'object_driven_G1_rmse_multi_out{output_size}_pred_end{predict_index_end}_{history}_{future}f_p{target_output}_skip1_num_G1_h1_fulledge_loss_all_{hidden_dim}_{voxel_size}'
    # model_prefix = f'rmse_multi_out{output_size}_pred_end{predict_index_end}_{history}_{future}f_p{target_output}_skip1_num_G1_h1_fulledge_loss_all_{hidden_dim}_{voxel_size}'
    model_prefix = f'multi2lr1e4_object{object_driven}_out{output_size}_pred_end{predict_index_end}_{history}_{future}f_p{target_output}_skip1_num_{hidden_dim}_G1_h1_fulledge_{hidden_dim}_{voxel_size}'

    print(model_prefix,history,future,p_start,p_end,voxel_size,num_nodes)




    train_x,train_y,test_x,test_y,val_x,val_y = get_train_test_data_on_users_all_videos(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
    print('shape of train_x:',train_x.shape,'shape of train_y:',train_y.shape,
          'shape of test_x:',test_x.shape,'shape of test_y:',test_y.shape,
          'shape of val_x:',val_x.shape,'shape of val_y:',val_y.shape)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    val_x = torch.from_numpy(val_x)
    val_y = torch.from_numpy(val_y)
    # import pdb;pdb.set_trace()
    # train_x[:,:,:,]
    
    train_dataset=torch.utils.data.TensorDataset(train_x,train_y)
    test_dataset=torch.utils.data.TensorDataset(test_x,test_y)
    val_dataset=torch.utils.data.TensorDataset(val_x,val_y)

    # test_dataset=torch.utils.data.TensorDataset(val_x,val_y)
    # val_dataset=torch.utils.data.TensorDataset(test_x,test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,num_workers=4,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=int(test_x.shape[0]/1),
                                            shuffle=False,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=int(val_x.shape[0]/1),
                                            shuffle=False,drop_last=True)     
    # test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                         batch_size=int(val_x.shape[0]),
    #                                         shuffle=False,drop_last=True)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                         batch_size=int(val_x.shape[0]),
    #                                         shuffle=False,drop_last=True)  
    # check test_loader and val_loader are same
    # import pdb;pdb.set_trace()
    # print('len of train_loader:',len(train_loader),'len of test_loader:',len(test_loader),'len of val_loader:',len(val_loader))  
    # load graph edges
    edge_prefix = str(voxel_size)
    edge_path = f'./data/{edge_prefix}/graph_edges_integer_index.csv'
    # r1, r2 = getedge('newedge',900)
    r1, r2 = getedge(edge_path)
    feature_num = train_x.shape[-1]
    assert feature_num == 7
    input_size = feature_num
    mymodel = GraphGRU(future,input_size,hidden_dim,output_size,history,num_nodes,r1,r2,batch_size)
    # if best model is saved, load it
    best_checkpoint_model_path = f'./data/model/best_model_{model_prefix}_checkpoint{history}_{future}.pt' 
    if os.path.exists(best_checkpoint_model_path):   
        mymodel.load_state_dict(torch.load(best_checkpoint_model_path))
        print(f'{best_checkpoint_model_path} model loaded')
    if torch.cuda.is_available():
        mymodel=mymodel.cuda()
    # print(mymodel)
    if with_train:
        # learning_rate=0.0003
        if predict_index_end==3:
            learning_rate = 0.0003  
            criterion = torch.nn.MSELoss()    # mean-squared error for regression
        else:
            learning_rate = 0.0003
            # criterion1 = torch.nn.MSELoss()    # mean-squared error for regression
            criterion = torch.nn.MSELoss()    # mean-squared error for regression
            # criterion = torch.nn.L1Loss()    # L1 loss
            # new loss using soft dice loss
            # criterion = SoftDiceLoss()
            # criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
        
        # optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate,weight_decay=0.01)
        optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer = torch.optim.AdamW(mymodel.parameters(), lr=learning_rate)
        lossa=[]
        val_loss_list = []

        # Initialize the early stopping object
        if continue_train_early_stop_val:
            early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=last_val_loss, path=best_checkpoint_model_path) #continue training the best check point
        else:
            early_stopping = EarlyStopping(patience=10, verbose=True, val_loss_min=float('inf'), path=best_checkpoint_model_path)
        # learning rate scheduler 
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, min_lr=1e-6)

        for epochs in range(1,num_epochs+1):
            mymodel.train()
            iter1 = 0
            iter2 = 0
            loss_total=0
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                # with record_function("model_training"):
            for i,(batch_x, batch_y) in tqdm(enumerate (train_loader)):
                # do batch normalization                 

                # batch_y_object = torch.zeros_like(batch_y) # (batch_size, self.output_window, self.num_nodes, self.output_dim)
                # batch_y_object[:,:,:,0] = batch_y[:,:,:,0]
                # batch_y_object[:,:,:,3:6] = batch_y[:,:,:,3:6]
                batch_y_object = batch_y.clone()

                batch_y=batch_y[:,-target_output,:,:]
                # keep batch_y as (batch_size, 1, self.num_nodes, self.output_dim)
                batch_y = batch_y.unsqueeze(1)  

                if torch.cuda.is_available():
                    batch_x=batch_x.cuda()
                
                    batch_y=batch_y.cuda() # (batch_size, self.output_window, self.num_nodes, self.output_dim)
                    batch_y_object = batch_y_object.cuda()
                    # zero like batch_y
                    
                # make sure we do not use future info by masking batch_y
                # import pdb;pdb.set_trace()
                if object_driven:
                # outputs = mymodel.forward_object(batch_x,batch_y_object) # (batch_size, self.output_window, self.num_nodes, self.output_dim)
                    outputs = mymodel.forward_output1_o(batch_x,batch_y_object)
                else:
                    outputs = mymodel(batch_x)
                optimizer.zero_grad()
                # break
                # import pdb;pdb.set_trace()

                # only get loss on the node who has points, in other words, the node whose occupancy is not 0
                # get the mask of the node whose occupancy is not 0, occupancy is the first feature in batch_y
                # ---------
                if predict_index_end==3:
                    outputs,batch_y = mask_outputs_batch_y(outputs, batch_y,output_size,predict_index_end)
                else:
                    batch_y = batch_y[:,:,:,predict_index_end-output_size:predict_index_end] # (batch_size, self.output_window, self.num_nodes, output_size)
                # ---------
                # import pdb;pdb.set_trace()
                # outputs = outputs.view(-1,num_nodes)
                # batch_y = batch_y.view(-1,num_nodes)

                # outputs = outputs.squeeze(3).squeeze(1)
                # batch_y = batch_y.squeeze(3).squeeze(1)

                loss = criterion(outputs,batch_y)
                loss_total=loss_total+loss.item()
                #backpropagation
                loss.backward()

                # Clip gradients
                # torch.nn.utils.clip_grad_norm_(mymodel.parameters(), max_norm=1)

                optimizer.step()
                iter1+=1
                # print loss
                if i % 100 == 0:
                    print("epoch:%d,  loss: %1.5f" % (epochs, loss.item()),flush=True)
                    # print(criterion1(outputs,batch_y).item())
            # Print profiler results
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))  
            # break                          
                
            loss_avg = loss_total/iter1
            losss=loss_avg
            lossa.append(losss)
            print("epoch:%d,  loss: %1.5f" % (epochs, loss_avg),flush=True)
            # save model every 10 epochs and then reload it to continue training
            if epochs % 10 == 0:
                #save and reloasd
                torch.save(mymodel.state_dict(), f'./data/model/graphgru_{model_prefix}_{history}_{future}_{epochs}.pt')
                print('model saved')
            # val_loss = get_val_loss(mymodel,val_loader,criterion,output_size)
            val_loss = get_val_loss(mymodel,val_loader,criterion,output_size,target_output,predict_index_end,object_driven=object_driven)
            val_loss_list.append(val_loss)
            print("val_loss:%1.5f" % (val_loss))

            # check the tesing loss for debug
            test_loss = get_val_loss(mymodel,test_loader,criterion,output_size,target_output,predict_index_end,object_driven=object_driven)
            print("test_loss:%1.5f" % (test_loss))


            # Step the scheduler with the validation loss
            scheduler.step(val_loss)  
            # Log the last learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr}')    
            # Call early stopping
            early_stopping(val_loss, mymodel)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        np.save(f'./data/output/graphgru_{model_prefix}_training_loss{history}_{future}',lossa)
        np.save(f'./data/output/graphgru_{model_prefix}_val_loss{history}_{future}',val_loss_list)
        print('loss saved')
        # plot training and val loss and save to file
        plt.figure()
        plt.plot(lossa)
        plt.plot(val_loss_list)
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'./data/fig/graphgru_{model_prefix}_trainingloss{history}_{future}.png')

    mymodel.load_state_dict(torch.load(best_checkpoint_model_path))



    with torch.no_grad():
        # train_x_nn,train_y_nn,test_x_nn,test_y_nn,val_x_nn,val_y_nn = get_train_test_data_on_users_all_videos_no_norm(history,future,p_start=p_start,p_end=p_end,voxel_size=voxel_size,num_nodes=num_nodes)
        # test_x_nn = torch.from_numpy(test_x_nn)
        # test_y_nn = torch.from_numpy(test_y_nn)
        # test_dataset_nn=torch.utils.data.TensorDataset(test_x_nn,test_y_nn)
        # test_loader_nn = torch.utils.data.DataLoader(dataset=test_dataset_nn,
        #                                         batch_size=int(test_x_nn.shape[0]/10),
        #                                         shuffle=False,drop_last=True)
        eval_model_sample(mymodel,test_loader,model_prefix,output_size,history=history,future=future,target_output=target_output,predict_index_end=predict_index_end)   
        # eval_model_sample_num(mymodel,test_loader,test_loader_nn,model_prefix,output_size,history=history,future=future)
        # eval_model(mymodel,test_loader,model_prefix,history=history,future=future)

if __name__ == '__main__':
    for future in [150,60,30,10]:
    # for future in [60]:
        print(f'future:{future}')
        main(future)
