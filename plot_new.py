import matplotlib.pyplot as plt
import numpy as np



# fsvvd
# Data
def fsvvd_mse():
    # Data
    # x = [1,10, 30, 60, 150]
    x = [33,333,1000,2000,5000]
    # mlp = [0.0280, 0.0735, 0.1080, 0.1818]
    # tlr = [0.026, 0.079, 0.138, 0.195]
    our = [0.00075,0.0058, 0.0127, 0.0158, 0.0182]
    # lr = [0.0422, 0.0966, 0.156, 0.192]
    lstm = [0.00036,0.0063, 0.0194, 0.0265, 0.0288]

    # Plotting with enhanced visualization
    plt.figure(figsize=(10, 6))

    plt.plot(x, our, label='Ours', linestyle='--', marker='o', markersize=8)
    # plt.plot(x, lr, label='LR', linestyle='--', marker='s', markersize=8)
    # plt.plot(x, tlr, label='TLR', linestyle='--', marker='D', markersize=8)
    # plt.plot(x, mlp, label='MLP', linestyle='--', marker='^', markersize=8)
    plt.plot(x, lstm, label='LSTM', linestyle='--', marker='^', markersize=8)




    plt.legend(title="Method", fontsize=16)
    plt.xlabel('Prediction Horizon (ms)', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    # plt.title('MSE Loss for  Prediction Across Different Horizons', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('../result/mse_loss.png')
    # plt.ylim(0, 0.22)

    # plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Values
def test_variance():

    # Data
    x_values = [1, 10, 30]  # X-axis values
    mse_values = [0.1, 0.2, 0.3]  # MSE values
    variance_values = [0.01, 0.02, 0.03]  # Variance of squared errors

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, mse_values, yerr=variance_values, fmt='-o', capsize=5, color='skyblue', ecolor='gray', elinewidth=2)
    plt.xlabel('X-axis')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE with Variance Error Bars')


    # Show plot
    plt.savefig('../result/mse_variance.png')

def fsvvd_r2():
    x = [1,10, 30, 60, 150]
    x = [33,333,1000,2000,5000]
    our = [0.9752,0.8079,0.5804,0.4897,0.4264]

    lstm = [0.9881,0.7925,0.3622,0.1447,0.0946]


    # Plotting with enhanced visualization
    plt.figure(figsize=(10, 6))

    plt.plot(x, our, label='Ours', linestyle='--', marker='o', markersize=8)
    # plt.plot(x, lr, label='LR', linestyle='--', marker='s', markersize=8)
    # plt.plot(x, tlr, label='TLR', linestyle='--', marker='D', markersize=8)
    # plt.plot(x, mlp, label='MLP', linestyle='--', marker='^', markersize=8)
    plt.plot(x, lstm, label='LSTM', linestyle='--', marker='^', markersize=8)




    plt.legend(title="Method", fontsize=16)
    plt.xlabel('Prediction Horizon (ms)', fontsize=18)
    plt.ylabel('R^2', fontsize=18)
    # R^2
    plt.title('R^2 Loss for  Prediction Across Different Horizons', fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim(0, 0.03)

    plt.savefig('../result/r2_loss.png')

if __name__ == '__main__':
    # fsvvd_mse()
    # test_variance()
    fsvvd_r2()