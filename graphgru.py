#!/bin/env python
from tqdm import tqdm
import os
from utils_graphgru import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from graphgru_model import *
from graphgru_eval import *

# torch.autograd.set_detect_anomaly(True)


# torch.set_default_tensor_type(torch.DoubleTensor)
# set to float32
# torch.set_default_dtype(torch.float32)
# torch.set_default_device()
# torch.set_default_tensor_type(torch.FloatTensor)
def main(future=10):
    with_train = True
    continue_train_early_stop_val = False
    user_previous_model = False # whether to load previous model
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
    predict_index_end=3 # 3 is occlusion, 2 is in-fov
    num_epochs=30
    batch_size = 32*4
    # batch_size=16 #multi_out
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
    model_prefix = f'multi2lr1e4_object_t1_g_only{object_driven}_out{output_size}_pred_end{predict_index_end}_{history}_{future}f_p{target_output}_skip1_num_{hidden_dim}_G1_h1_fulledge_{hidden_dim}_{voxel_size}'

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
    if user_previous_model:
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
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, min_lr=1e-6)

        for epochs in range(1,num_epochs+1):
            mymodel.train()
            iter1 = 0
            iter2 = 0
            loss_total=0
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                # with record_function("model_training"):
            for i,(batch_x, batch_y) in tqdm(enumerate (train_loader)):
                # do batch normalization                 
                # import pdb;pdb.set_trace()
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
    # for future in [60,30,10]:
        print(f'future:{future}')
        main(future)
