import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

def plot_loss():
    # read graphgruloss.txt.npy from ./data/
    loss = np.load('./data/graphgruloss.txt.npy')
    print(loss)
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphGRU Loss')
    plt.show()

def plot_test_loss():
    test_loss = '''TIME:1 ,MAE:0.01100,  MAPE: 4967.57420, MSE: 0.00120
TIME:2 ,MAE:0.01252,  MAPE: 5561.65457, MSE: 0.00167
TIME:3 ,MAE:0.01287,  MAPE: 5275.02838, MSE: 0.00209
TIME:4 ,MAE:0.01355,  MAPE: 5290.98404, MSE: 0.00246
TIME:5 ,MAE:0.01434,  MAPE: 5475.79434, MSE: 0.00281
TIME:6 ,MAE:0.01504,  MAPE: 5592.45853, MSE: 0.00315
TIME:7 ,MAE:0.01566,  MAPE: 5655.26026, MSE: 0.00348
TIME:8 ,MAE:0.01635,  MAPE: 5769.01973, MSE: 0.00381
TIME:9 ,MAE:0.01722,  MAPE: 6004.37472, MSE: 0.00413
TIME:10 ,MAE:0.01794,  MAPE: 6149.35540, MSE: 0.00444
TIME:11 ,MAE:0.01859,  MAPE: 6253.38856, MSE: 0.00475
TIME:12 ,MAE:0.01925,  MAPE: 6371.78862, MSE: 0.00505
TIME:13 ,MAE:0.01994,  MAPE: 6516.00562, MSE: 0.00535
TIME:14 ,MAE:0.02067,  MAPE: 6699.71930, MSE: 0.00565
TIME:15 ,MAE:0.02133,  MAPE: 6835.53969, MSE: 0.00594
TIME:16 ,MAE:0.02198,  MAPE: 6983.69232, MSE: 0.00622
TIME:17 ,MAE:0.02261,  MAPE: 7115.62038, MSE: 0.00650
TIME:18 ,MAE:0.02322,  MAPE: 7244.47402, MSE: 0.00678
TIME:19 ,MAE:0.02388,  MAPE: 7424.93449, MSE: 0.00705
TIME:20 ,MAE:0.02447,  MAPE: 7545.64782, MSE: 0.00732
TIME:21 ,MAE:0.02504,  MAPE: 7664.56497, MSE: 0.00759
TIME:22 ,MAE:0.02563,  MAPE: 7803.59219, MSE: 0.00785
TIME:23 ,MAE:0.02623,  MAPE: 7950.74163, MSE: 0.00811
TIME:24 ,MAE:0.02682,  MAPE: 8106.68563, MSE: 0.00836
TIME:25 ,MAE:0.02741,  MAPE: 8271.45230, MSE: 0.00861
TIME:26 ,MAE:0.02796,  MAPE: 8398.91933, MSE: 0.00886
TIME:27 ,MAE:0.02852,  MAPE: 8542.96434, MSE: 0.00910
TIME:28 ,MAE:0.02904,  MAPE: 8658.68878, MSE: 0.00934
TIME:29 ,MAE:0.02957,  MAPE: 8798.29476, MSE: 0.00957
TIME:30 ,MAE:0.03007,  MAPE: 8909.57344, MSE: 0.00980'''
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
    plt.show()


if __name__ == '__main__':
    # plot_loss()
    plot_test_loss()