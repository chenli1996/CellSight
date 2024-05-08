import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

def plot_loss():
    # read graphgruloss.txt.npy from ./data/
    loss = np.load('./data/graphgruloss150_60.txt.npy')
    print(loss)
    # plt.plot(np.log(loss))
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphGRU Loss')
    # plt.show()
    plt.savefig('./data/fig/graphgru_training_loss_90_60.png')

    loss_string = '''133it [05:01,  2.27s/it]
epoch:0,  loss: 0.02686
model saved
133it [04:51,  2.19s/it]
epoch:1,  loss: 0.02256
133it [04:51,  2.19s/it]
epoch:2,  loss: 0.02225
133it [04:52,  2.20s/it]
epoch:3,  loss: 0.02205
133it [04:50,  2.18s/it]
epoch:4,  loss: 0.02188
133it [04:50,  2.19s/it]
epoch:5,  loss: 0.02170
133it [04:50,  2.19s/it]
epoch:6,  loss: 0.02151
133it [04:51,  2.19s/it]
epoch:7,  loss: 0.02135
133it [04:50,  2.19s/it]
epoch:8,  loss: 0.02115
133it [04:51,  2.19s/it]
epoch:9,  loss: 0.02110
133it [04:51,  2.19s/it]
epoch:10,  loss: 0.02098
model saved
133it [04:50,  2.18s/it]
epoch:11,  loss: 0.02090
133it [04:51,  2.19s/it]
epoch:12,  loss: 0.02079
133it [04:51,  2.19s/it]
epoch:13,  loss: 0.02080
133it [04:52,  2.20s/it]
epoch:14,  loss: 0.02074
133it [04:50,  2.18s/it]
epoch:15,  loss: 0.02073
133it [04:49,  2.18s/it]
epoch:16,  loss: 0.02064
133it [04:50,  2.19s/it]
epoch:17,  loss: 0.02093
133it [04:50,  2.18s/it]
epoch:18,  loss: 0.02092
133it [04:51,  2.19s/it]
epoch:19,  loss: 0.02047
133it [05:06,  2.30s/it]
epoch:20,  loss: 0.02046
model saved
133it [04:53,  2.20s/it]
epoch:21,  loss: 0.02043
133it [04:47,  2.16s/it]
epoch:22,  loss: 0.02038
133it [04:44,  2.14s/it]
epoch:23,  loss: 0.02035
133it [04:42,  2.13s/it]
epoch:24,  loss: 0.02031
133it [04:41,  2.11s/it]
epoch:25,  loss: 0.02036
133it [04:41,  2.12s/it]
epoch:26,  loss: 0.02047
133it [04:41,  2.12s/it]
epoch:27,  loss: 0.02028
133it [04:41,  2.12s/it]
epoch:28,  loss: 0.02020
133it [04:42,  2.13s/it]
epoch:29,  loss: 0.02014
133it [04:41,  2.12s/it]
epoch:30,  loss: 0.02010
model saved
133it [04:42,  2.12s/it]
epoch:31,  loss: 0.02014
133it [04:47,  2.16s/it]
epoch:32,  loss: 0.02016
133it [04:42,  2.12s/it]
epoch:33,  loss: 0.02013
133it [04:41,  2.11s/it]
epoch:34,  loss: 0.02011
133it [04:40,  2.11s/it]
epoch:35,  loss: 0.02022
133it [04:41,  2.11s/it]
epoch:36,  loss: 0.02013
133it [04:42,  2.12s/it]
epoch:37,  loss: 0.01990
133it [04:42,  2.12s/it]
epoch:38,  loss: 0.01979
133it [04:41,  2.12s/it]
epoch:39,  loss: 0.01975
133it [04:42,  2.13s/it]
epoch:40,  loss: 0.01979
model saved
133it [04:41,  2.12s/it]
epoch:41,  loss: 0.01986
133it [04:41,  2.12s/it]
epoch:42,  loss: 0.01980
133it [04:42,  2.12s/it]
epoch:43,  loss: 0.01969
133it [05:07,  2.31s/it]
epoch:44,  loss: 0.01972
133it [04:41,  2.12s/it]
epoch:45,  loss: 0.01961
133it [04:41,  2.11s/it]
epoch:46,  loss: 0.01960
133it [04:42,  2.13s/it]
epoch:47,  loss: 0.01952
133it [04:41,  2.12s/it]
epoch:48,  loss: 0.01952
133it [04:41,  2.12s/it]
epoch:49,  loss: 0.01951
133it [04:41,  2.12s/it]
epoch:50,  loss: 0.01952
model saved
133it [04:42,  2.12s/it]
epoch:51,  loss: 0.01950
133it [04:41,  2.12s/it]
epoch:52,  loss: 0.01953
133it [04:42,  2.12s/it]
epoch:53,  loss: 0.01949
133it [05:21,  2.42s/it]
epoch:54,  loss: 0.01942
133it [04:42,  2.13s/it]
epoch:55,  loss: 0.01932
133it [04:41,  2.12s/it]
epoch:56,  loss: 0.01929
133it [04:42,  2.12s/it]
epoch:57,  loss: 0.01919
133it [04:40,  2.11s/it]
epoch:58,  loss: 0.01926
133it [04:42,  2.12s/it]
epoch:59,  loss: 0.01927
133it [04:41,  2.12s/it]
epoch:60,  loss: 0.01921
model saved
133it [04:41,  2.11s/it]
epoch:61,  loss: 0.01916
133it [04:41,  2.12s/it]
epoch:62,  loss: 0.01905
133it [04:42,  2.12s/it]
epoch:63,  loss: 0.01902
133it [04:42,  2.12s/it]
epoch:64,  loss: 0.01912
133it [04:41,  2.12s/it]
epoch:65,  loss: 0.01921
133it [04:41,  2.12s/it]
epoch:66,  loss: 0.01938
133it [04:40,  2.11s/it]
epoch:67,  loss: 0.01928
133it [04:41,  2.12s/it]
epoch:68,  loss: 0.01907
133it [04:43,  2.13s/it]
epoch:69,  loss: 0.01899
133it [04:41,  2.12s/it]
epoch:70,  loss: 0.01879
model saved
133it [04:41,  2.12s/it]
epoch:71,  loss: 0.01871
133it [04:41,  2.12s/it]
epoch:72,  loss: 0.01915
133it [04:41,  2.12s/it]
epoch:73,  loss: 0.01940
133it [04:40,  2.11s/it]
epoch:74,  loss: 0.01910
133it [04:41,  2.11s/it]
epoch:75,  loss: 0.01900
133it [04:46,  2.15s/it]
epoch:76,  loss: 0.01878
133it [04:41,  2.12s/it]
epoch:77,  loss: 0.01860
133it [04:40,  2.11s/it]
epoch:78,  loss: 0.01853
133it [04:42,  2.13s/it]'''
    loss_string = loss_string.split('\n')
    loss_string = [x.split(',') for x in loss_string if x[0] == 'e']
    # import pdb; pdb.set_trace()
    loss_mse = [x[1].split(':') for x in loss_string]
    loss_mse = [float(x[1]) for x in loss_mse]
    print(loss_mse)
    plt.plot(loss_mse, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphGRU Loss')
    plt.savefig('./data/fig/graphgru_training_loss.png')


def plot_test_loss(test_loss_list):
    for test_loss in test_loss_list:
        test_loss = test_loss.split('\n')
        test_loss = [x.split(',') for x in test_loss]
        test_loss_mae = [x[1].split(':') for x in test_loss]
        test_loss_mae = [float(x[1]) for x in test_loss_mae]
        print(test_loss_mae)
        plt.plot(test_loss_mae, label='MAE')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title('GraphGRU Test Loss')

        test_loss_mse = [x[3].split(':') for x in test_loss]
        test_loss_mse = [float(x[1]) for x in test_loss_mse]
        print(test_loss_mse)
        plt.plot(test_loss_mse, label='MSE')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title('GraphGRU Test Loss')
        plt.legend()
        # plt.show()
    plt.savefig('./data/fig/graphgru_test_loss.png')




    plt.savefig('./data/fig/graphgru_test_loss_90_60.png')


def plot_loss_with_some_points():
    MSE = [0.0011824649991467595, 0.0015528767835348845, 0.0019689053297042847, 0.0024274634197354317, 0.002800989430397749, 0.0031833541579544544, 0.0035882091615349054, 0.003919435199350119, 0.00428751902654767, 0.004600231070071459]
    # MAE =[0.014175975695252419, 0.015277200378477573, 0.0164263304322958, 0.017489546909928322, 0.01854783296585083, 0.01942448690533638, 0.02040746808052063, 0.021344339475035667, 0.02215290628373623, 0.023020528256893158, 0.02368580363690853, 0.024312060326337814, 0.02503584884107113, 0.025686252862215042, 0.026249995455145836, 0.026760172098875046, 0.027317438274621964, 0.02794686332345009, 0.028426695615053177, 0.028948776423931122, 0.029385995119810104, 0.02984929457306862, 0.03021187335252762, 0.030622657388448715, 0.031150463968515396, 0.03154768794775009, 0.03188407048583031, 0.03224803879857063, 0.032401420176029205, 0.03276808187365532, 0.033102016896009445, 0.0333375446498394, 0.03372472524642944, 0.03379673883318901, 0.03406703844666481, 0.034022215753793716, 0.03438208997249603, 0.03442130237817764, 0.03466606140136719, 0.03489881753921509, 0.03512019291520119, 0.035352129489183426, 0.03527214750647545, 0.03538960963487625, 0.03558524325489998, 0.035626672208309174, 0.03584333881735802, 0.035903241485357285, 0.035809360444545746, 0.035924267023801804, 0.036105405539274216, 0.03614910691976547, 0.03598684445023537, 0.035937488079071045, 0.036114636808633804, 0.03606199100613594, 0.03595395013689995, 0.03617316111922264, 0.036025095731019974, 0.03573495149612427]
    plt.plot(MSE, label='model-h90 MSE')
    # plt.plot(MAE, label='model-90 MAE')
    # plot some points
    Loss_list = [0.004,0.007,0.0129,0.0201]
    x_list = [1,10,30,60]

    Loss_list_30 = [0.0014,0.0043,0.0102,0.0173]
    Loss_list_10 = [0.0003,0.0027,0.0085,0.0158]

    MAE_LR_90 = [0.01145,0.01607,0.0251,0.0354]
    # plot with dot mark
    plt.plot(x_list, Loss_list, 'o', label='LR-h90 MSE')
    plt.plot(x_list, Loss_list_30, 'o', label='LR-h30 MSE')
    # plt.plot(x_list, Loss_list_10, 'o')
    # plt.plot(x_list, MAE_LR_90, 'x', label='LR-h90 MAE')
    plt.xlabel('Prediction Horizon/frame')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./data/fig/graphgru_loss_withLR_30_10.png')


if __name__ == '__main__':

    
    # plot_loss()
    # plot_loss_from_string()
    # plot_test_loss(test_loss_list=[test_lost_list[-1]])
    plot_loss_with_some_points()