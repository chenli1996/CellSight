
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import matplotlib.pyplot as plt
from utils_graphgru import *
from sklearn.metrics import r2_score

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
    r2_score_list = []
    MAE = {}
    MAPE = {}
    MSE = {}
    RMSE = {}
    R2_scre = {}
    # criterion = torch.nn.MSELoss()    

    for i in range(future):
        MAE[i] = 0
        MAPE[i] = 0
        MSE[i] = 0
        RMSE[i] = 0
        R2_scre[i] = 0

    with torch.no_grad():
        # import pdb;pdb.set_trace()

        for i,(batch_x, batch_y) in enumerate (test_loader):
            assert i == 0 # batch size is equal to the test set size

            # import pdb;pdb.set_trace()
            # only predict last frame------
            batch_y=batch_y[:,-target_output,:,:]
            # keep batch_y as (batch_size, 1, self.num_nodes, self.output_dim)
            batch_y = batch_y.unsqueeze(1) 
            # ----------------------------- 

            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            outputs = net(batch_x) # (batch_size, self.output_window, self.num_nodes, self.output_dim)

    # -------------------mask the output and label to get the non-zero values
            # outputs = outputs.unsqueeze(1).unsqueeze(-1)
            # batch_y = batch_y.unsqueeze(1).unsqueeze(-1)
            # batch_y = batch_y.unsqueeze(1)
            # import pdb;pdb.set_trace()
            outputs_copy = outputs.repeat(1,future,1,7).clone()
            batch_y_copy = batch_y.repeat(1,future,1,1).clone()
            # import pdb;pdb.set_trace()
            pred_y_o_non_zero,test_y_o_non_zero = mask_outputs_batch_y_cut(outputs_copy,batch_y_copy, future-1, output_size, predict_index_end)
            # import pdb;pdb.set_trace()
            # pred_y_o_non_zero = pred_y_o_non_zero.cpu().detach().numpy()
            # test_y_o_non_zero = test_y_o_non_zero.cpu().detach().numpy()
            mse_non_zero = mse(pred_y_o_non_zero, test_y_o_non_zero).cpu().detach().numpy()
            print(f'MSE_non_zero-o:{mse_non_zero}')
            std_squared_errors_non_zero = torch.std((pred_y_o_non_zero - test_y_o_non_zero) ** 2)
            print(f'std_squared_errors_non_zero:{std_squared_errors_non_zero}')
            # import pdb;pdb.set_trace()
            r2_score_non_zero = r2_score(test_y_o_non_zero.cpu().detach(), pred_y_o_non_zero.cpu().detach())
            print(f'R² Score (Non-Zero): {r2_score_non_zero}')
            del outputs_copy,batch_y_copy,pred_y_o_non_zero,test_y_o_non_zero
    # -------------------    



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
                R2_score_d = r2_score(batch_y[:, u, :, :].contiguous().cpu().detach().view(-1).numpy(),outputs[:, u, :, :].contiguous().cpu().detach().view(-1).numpy())
                y_pred = outputs[:, u, :, :].contiguous().cpu().detach().view(-1)
                y_true = batch_y[:, u, :, :].contiguous().cpu().detach().view(-1)
                squared_errors = (y_true - y_pred) ** 2
                std_squared_errors = torch.std(squared_errors)

                # y_true_non_zero is the true values that are not zero
                y_true_non_zero = y_true[y_true != 0]
                y_pred_non_zero = y_pred[y_true != 0]
                squared_errors_non_zero = (y_true_non_zero - y_pred_non_zero) ** 2
                std_squared_errors_non_zero = torch.std(squared_errors_non_zero)
                mse_non_zero = mse(y_pred_non_zero, y_true_non_zero).cpu().detach().numpy()
                print(f'std_squared_errors_non_zero:{std_squared_errors_non_zero}')
                print(f'mse_non_zero-v:{mse_non_zero}')
                print(f'std_squared_errors:{std_squared_errors}')
                # import pdb;pdb.set_trace()


                MAE[u] += MAE_d
                MAPE[u] += MAPE_d
                MSE[u] += MSE_d
                RMSE[u] += RMSE_d
                R2_scre[u] += R2_score_d
        for u in range(outputs.shape[1]):
            MAE_u = MAE[u]/(i+1)
            MAPE_u = MAPE[u]/(i+1)
            MSE_u = MSE[u]/(i+1)
            RMSE_u = RMSE[u]/(i+1)
            R2_scre_u = R2_scre[u]/(i+1)
            # print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f, RMSE: %1.5f" % ((u+1),MAE_u, MAPE_u,MSE_u,RMSE_u))
            # print("TIME:%d ,MAE:%1.5f,  MAPE: %1.5f, MSE: %1.5f, RMSE: %1.5f, R2_score: %1.5f" % ((u+1),MAE_u, MAPE_u,MSE_u,RMSE_u,R2_scre_u))
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
            r2_score_list.append(R2_scre_u)

        print('MSE:',mse_list)
        print('MAE:',mae_list)
        print('RMSE:',rmse_list)
        print('R2_score:',r2_score_list)

    
        # print('MAPE:',mape_list)
        # plot mse and mae
        # plt.figure()
        # if len(mse_list) == 1:
        #     plt.scatter(future, mse_list[0])
        # else:
        #     plt.plot(mse_list)

        # if len(mae_list) == 1:
        #     plt.scatter(future, mae_list[0])
        # else:
        #     plt.plot(mae_list)
        # # plt.plot(mape_list)
        # plt.legend(['MSE', 'MAE'])
        # plt.xlabel('Prediction Horizon/frame')
        # plt.ylabel('Loss')
        # plt.savefig(f'./data/fig/p90_vs128_graphgru_{model_prefix}_testingloss{history}_{future}.png') 


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
    
