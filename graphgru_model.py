import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.data import Data, Batch
import numpy as np


# torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)
# torch.set_default_device('cpu')  # or 'cuda' if you're using a GPU
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
            # torch.FloatTensor(self._num_gru_units + self._num_gru_units, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()
        self.num_nodes = num_nodes

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    # def forward(self, inputs, hidden_state):
    #     batch_size = hidden_state.shape[0]
    #     # assert batch_size == 200
    #     inputs = inputs.reshape((batch_size, self.num_nodes, self.feature_num))
    #     # inputs (batch_size, num_nodes, feature_num)
    #     # hidden_state (batch_size, num_nodes, num_gru_units)
    #     hidden_state = hidden_state.reshape(
    #         (batch_size, self.num_nodes, self._num_gru_units)
    #     )
    #     # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
    #     # import pdb;pdb.set_trace()
    #     concatenation = torch.cat((inputs, hidden_state), dim=2)
    #     # [x, h] (batch_size * num_nodes, gru_units + 1)
    #     concatenation = concatenation.reshape((-1, self._num_gru_units + self.feature_num))
    #     # [x, h]W + b (batch_size * num_nodes, output_dim)
    #     outputs = concatenation @ self.weights + self.biases
    #     # [x, h]W + b (batch_size, num_nodes, output_dim)
    #     outputs = outputs.reshape((batch_size, self.num_nodes, self._output_dim))
    #     # [x, h]W + b (batch_size, num_nodes * output_dim)
    #     #outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
    #     return outputs
    
    def forward(self, inputs, hidden_state):
        batch_size = hidden_state.shape[0]
        # Reshape inputs and hidden_state to have explicit dimensions
        inputs = inputs.reshape((batch_size, self.num_nodes, -1))
        # inputs = inputs.reshape((batch_size, self.num_nodes, self.feature_num))
        hidden_state = hidden_state.reshape((batch_size, self.num_nodes, self._num_gru_units))
        # import pdb;pdb.set_trace()
        # Concatenate inputs and hidden_state along the feature dimension
        concatenation = torch.cat((inputs, hidden_state), dim=2)  # Shape: (batch_size, num_nodes, feature_num + num_gru_units)
        # import pdb;pdb.set_trace()
        # Perform batched matrix multiplication directly on the 3D tensor
        outputs = concatenation @ self.weights + self.biases  # Shape: (batch_size, num_nodes, output_dim)

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
        # self.edge_index_expanded = self.precompute_edge_index_sparse(self.batch_size)

        self.head = 1
        self.multiGAT = False
        self.dropout = 0.2
        self.OriginalGAT = False
        # self.GCN3 = GATConv(self.num_units+self.input_dim, self.num_units)
        if self.OriginalGAT:
            # for [S,G]
            # self.GCN3 = GATConv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            # for G only
            self.GCN3 = GATConv(self.input_dim, self.num_units,heads=self.head,concat=False)
            # for S only

            self.GCN4 = GATConv(self.num_units,self.num_units,concat=False)
        else:
            # self.GAT3 = GATv2Conv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            # self.GAT4 = GATv2Conv(self.num_units*self.head,self.num_units,concat=False)

            # for [S,G]
            self.GAT3 = TransformerConv(self.num_units+self.input_dim, self.num_units,heads=self.head,concat=False)
            # for G only
            # self.GAT3 = TransformerConv(self.input_dim, self.num_units,heads=self.head,concat=False)
            # for G and S individually
            # self.GAT_state = TransformerConv(self.num_units, self.num_units,concat=False)
            # for S only
            # self.GAT3 = TransformerConv(self.num_units, self.num_units,heads=self.head,concat=False)



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

    def precompute_edge_index_sparse(self, batch_size=None,x=None):
        import pdb;pdb.set_trace()
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
        # # original graph with gru----------------------------------------------
        # # inputs (batch_size, num_nodes * input_dim)
        # # state (batch_size, num_nodes * gru_units) or (batch_size, num_nodes* num_units)
        # batch_size = state.shape[0]
        # # import pdb;pdb.set_trace()
        # # update state using graph neighbors
        state=self._gc3(state,inputs, self.num_units) # (batch_size, self.num_nodes * self.gru_units)
        output_size = 2 * self.num_units
        value = torch.sigmoid(
            self.GRU1(inputs, state))  # (batch_size, self.num_nodes, output_size)
        # import pdb;pdb.set_trace()
        reset, z_var = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1) # (batch_size, self.num_nodes, self.gru_units)
        reset = torch.reshape(reset, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        z_var = torch.reshape(z_var, (-1, self.num_nodes * self.num_units))
        c = self.act(self.GRU2(inputs, reset * state))
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        # new_state = z_var * state + (1.0 - z_var) * c
        new_state = (1.0 - z_var) * state + z_var * c
        # -----------------------------------------------------------------------



        # graph on input only----------------------------------------------
        # state is h_{t-1}, inputs is x_t
        # update state using graph neighbors
        # inputs=self._gc3(state,inputs, self.num_units) # (batch_size, self.num_nodes * self.gru_units)
        # state = self._gc3_state(state, self.num_units) # (batch_size, self.num_nodes * self.gru_units)
        # output_size = 2 * self.num_units
        # value = torch.sigmoid(
        #     self.GRU1(inputs, state))  # (batch_size, self.num_nodes, output_size)
        # # import pdb;pdb.set_trace()
        # reset, z_var = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1) # (batch_size, self.num_nodes, self.gru_units)
        # reset = torch.reshape(reset, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        # z_var = torch.reshape(z_var, (-1, self.num_nodes * self.num_units))  #should we separate all nodes? like using z_var = z_var.view(-1, self.num_units)
        # c = self.act(self.GRU2(inputs, reset * state))
        # c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        # # new_state = z_var * state + (1.0 - z_var) * c
        # new_state = (1.0 - z_var) * state + z_var * c




        # -----------------------------------------------------------------------



        return new_state

    def _gc3_state(self, state, output_size, bias_start=0.0):
        batch_size = state.shape[0]
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))
        x = state.to(self.device)
        batch_size, num_nodes, num_features = x.size()
        x_flat = x.reshape(-1, num_features)  # (batch_size * num_nodes, input_dim)
        if batch_size == self.batch_size:
            edge_index = self.edge_index_expanded
        else:
            edge_index = self.precompute_edge_index(batch_size)
        data = Data(x=x_flat, edge_index=edge_index)
        batch = Batch.from_data_list([data])

        x = self.GAT_state(batch.x, batch.edge_index)
        x = x.reshape(shape=(batch_size, self.num_nodes * output_size))
        return x
    
    def remove_nodes_and_edges(self,data):
        x, edge_index = data.x, data.edge_index

        # Step 1: Identify nodes to keep
        nodes_to_keep_mask = x[:, -7] != 0  # Boolean mask of nodes to keep
        nodes_to_keep = nodes_to_keep_mask.nonzero(as_tuple=False).view(-1)  # Indices of nodes to keep

        # Step 2: Create a mapping from old node indices to new node indices
        new_indices = -torch.ones(x.size(0), dtype=torch.long, device=x.device)
        new_indices[nodes_to_keep] = torch.arange(nodes_to_keep.size(0), device=x.device)

        # Step 3: Filter edges where both nodes are kept
        edge_mask = nodes_to_keep_mask[edge_index[0]] & nodes_to_keep_mask[edge_index[1]]
        edge_index_filtered = edge_index[:, edge_mask]

        # Map old node indices to new node indices
        edge_index_mapped = new_indices[edge_index_filtered]

        # Step 4: Adjust node features
        x_new = x[nodes_to_keep]

        # Step 5: Adjust the batch vector if it exists
        if hasattr(data, 'batch') and data.batch is not None:
            batch_new = data.batch[nodes_to_keep]
        else:
            batch_new = None

        # Step 6: Create a new Data object
        data_new = Data(x=x_new, edge_index=edge_index_mapped)
        if batch_new is not None:
            data_new.batch = batch_new

        # Copy other data attributes if necessary
        for key in set(data.keys()) - {'x', 'edge_index', 'batch'}:
            data_new[key] = data[key]

        # Store nodes_to_keep in data_new
        data_new.nodes_to_keep = nodes_to_keep
        return data_new




    def _gc3(self, state, inputs, output_size, bias_start=0.0):

        batch_size = state.shape[0]
        # assert batch_size == 200
        # import pdb;pdb.set_trace()

        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1)) # (batch, self.num_nodes, self.input_dim)
        # import pdb;pdb.set_trace()

        # original [S,G]-----------------
        inputs_and_state = torch.cat([state, inputs], dim=2)
        # input_size = inputs_and_state.shape[2]
        x = inputs_and_state.to(self.device)
        # G only-------------------------
        # x = inputs.to(self.device)
        # S only-------------------------
        # x = state.to(self.device)


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

        #  dynamic graph with gru----------------------------------------------
        # edge_index = self.precompute_edge_index_sparse(batch_size,x)
        # -----------------------------------------------------------------------


        # x_flat = x.view(-1, num_features)  # Shape: (batch_size * num_nodes, num_features)
        x_flat = x.reshape(-1, num_features)  # (batch_size * num_nodes, input_dim)

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
        #     edge_index = self.precompute_edge_index_sparse(batch_size)

# using dynamic graph and compute the edge_index for each time step
        # if batch_size == self.batch_size:
            # edge_index = self.precompute_edge_index_sparse(self.batch_size)
        
        
        # import pdb;pdb.set_trace()
        data = Data(x=x_flat, edge_index=edge_index)

        sparse_data = self.remove_nodes_and_edges(data)
        # import pdb;pdb.set_trace()
        batch = Batch.from_data_list([sparse_data])

        # batch = Batch.from_data_list([data])
        
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

        # import pdb;pdb.set_trace()

        # uniform graph----------------------------------------------
        # x = x.reshape(shape=(batch_size, self.num_nodes* output_size))
        # import pdb;pdb.set_trace()


# dynamic graph----------------------------------------------
#         # Step 3: Reconstruct the original tensor
        # Option 2: Keep original features for dropped nodes
        x_output = x_flat.clone()[:, :self.num_units]
        x_output[sparse_data.nodes_to_keep] = x
        x_output = x_output.reshape(shape=(batch_size, self.num_nodes* output_size))
        x = x_output

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

        # self.edge_index_expanded = self.precompute_edge_index(self.batch_size)
        self.edge_index_expanded = self.precompute_edge_index_sparse(self.batch_size)

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
        # import pdb;pdb.set_trace()
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
    
    def precompute_edge_index_sparse(self, batch_size=None):
        # import pdb;pdb.set_trace()
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
# using dynamic graph and compute the edge_index for each time step
        # edge_index = self.precompute_edge_index_sparse(batch_size)
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
    
