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

import matplotlib.pyplot as plt
import seaborn as sns

def plot_viewport_fsvvd(dataset,name):
    # Time resolutions
    time_ms = [33, 333, 1000, 2000, 5000]

    # MSE values for each model
    # viewport fsvvd
    if dataset == 'FSVVD':
        if name == 'Viewport Overlap Ratio':
            mlp_mse = [0.0025, 0.0569, 0.2035, 0.2656, 0.2996]
            tlr_mse = [0.0043, 0.0506, 0.1643, 0.2473, 0.3106]
            lr90_mse = [0.0733, 0.1113, 0.1746, 0.2331, 0.3034]
            lr30_mse = [0.0239, 0.0777, 0.1715, 0.2398, 0.3060]
            lstm_mse = [0.0010, 0.0446, 0.1693, 0.2379, 0.2696]
            ours_mse = [0.0011, 0.0332, 0.0947, 0.1256, 0.1490]
        elif name == 'Angular Span':
            lstm_mse = [0.0001, 0.0035, 0.0131, 0.0186, 0.0198]
            ours_mse = [0.0002, 0.0031, 0.0075, 0.0098, 0.0110]
            mlp_mse = [0.0003, 0.0045, 0.0154, 0.0189, 0.0217]
            tlr_mse = [0.0003, 0.0038, 0.0129, 0.0199, 0.0233]
            lr90_mse = [0.0057, 0.0089, 0.0140, 0.0183, 0.0233]
            lr30_mse = [0.0018, 0.0060, 0.0138, 0.0189, 0.0227]
        elif name == 'Occlusion-aware Visibility':
            # visibilty fsvvd
            lstm_mse = [0.00036, 0.0063, 0.0194, 0.0265, 0.0287]
            ours_mse = [0.00096, 0.0059, 0.0128, 0.0161, 0.0189]
            mlp_mse = [0.0007, 0.0074, 0.0230, 0.0251, 0.0318]
            tlr_mse = [0.0008, 0.0071, 0.0198, 0.0294, 0.0347]
            lr90_mse = [0.0096, 0.0143, 0.0218, 0.0288, 0.0333]
            lr30_mse = [0.0035, 0.0095, 0.0203, 0.0284, 0.0344]
        elif name == 'Visibile Angular Span':
            lstm_mse = [0.0001, 0.0009, 0.0021, 0.0027, 0.0031]
            ours_mse = [0.0001, 0.0008, 0.0013, 0.0017, 0.0020]
            mlp_mse = [0.0001, 0.0008, 0.0024, 0.0026, 0.0032]
            tlr_mse = [0.0001, 0.0008, 0.0018, 0.0024, 0.0028]
            lr90_mse = [0.0010, 0.0015, 0.0021, 0.0024, 0.0028]
            lr30_mse = [0.0004, 0.0010, 0.0018, 0.0024, 0.0027]
        else:
            print('name error',name)
            pass

    # # resolution fsvvd
    #     # LSTM model MSE values
    # lstm_mse = [0.00006, 0.00082, 0.00209, 0.00268, 0.0031]
    # # Ours model MSE values
    # ours_mse = [0.00021, 0.00078, 0.00134, 0.00167, 0.0020]
    # # MLP model MSE values
    # mlp_mse = [0.0001, 0.0008, 0.0024, 0.0026, 0.0032]
    # # TLR model MSE values
    # tlr_mse = [0.0001, 0.0008, 0.0018, 0.0024, 0.0028]
    # # LR90 model MSE values
    # lr90_mse = [0.001, 0.0015, 0.0021, 0.0024, 0.0028]
    # # LR30 model MSE values
    # lr30_mse = [0.0004, 0.001, 0.0018, 0.0024, 0.0027]
    elif dataset == '8i':
        if name == 'Viewport Overlap Ratio':
    # # viewport 8i
            lstm_mse = [0.0025, 0.0238, 0.068, 0.1117, 0.1394]
            ours_mse = [0.00103, 0.0158, 0.0569, 0.0705, 0.0842]
            mlp_mse = [0.0038, 0.0281, 0.0735, 0.108, 0.1818]
            tlr_mse = [0.004, 0.0267, 0.0791, 0.1381, 0.1954]
            lr90_mse = [0.0391, 0.0624, 0.11, 0.1674, 0.196]
            lr30_mse = [0.013, 0.0423, 0.0967, 0.1564, 0.1921]
        elif name == 'Angular Span':
            lstm_mse = [0.0011, 0.0021, 0.0054, 0.0087, 0.0135]
            ours_mse = [0.0001, 0.0015, 0.0038, 0.0060, 0.0074]
            mlp_mse = [0.0012, 0.0030, 0.0063, 0.0092, 0.0143]
            tlr_mse = [0.0005, 0.0020, 0.0056, 0.0105, 0.0151]
            lr90_mse = [0.0029, 0.0048, 0.0087, 0.0124, 0.0150]
            lr30_mse = [0.0008, 0.0026, 0.0065, 0.0110, 0.0148]
        elif name == 'Occlusion-aware Visibility':
            # visibilty 8i
             # MSE values for each model
            lstm_mse = [0.0004, 0.0032, 0.0081, 0.0131, 0.0161]
            ours_mse = [0.0005, 0.0044, 0.0098, 0.0113, 0.0117]
            mlp_mse = [0.0006, 0.0036, 0.0093, 0.0137, 0.0233]
            tlr_mse = [0.0006, 0.0029, 0.0086, 0.0158, 0.0224]
            lr90_mse = [0.0044, 0.0071, 0.0129, 0.0201, 0.0232]
            lr30_mse = [0.0015, 0.0043, 0.0102, 0.0174, 0.0229]
        elif name == 'Visibile Angular Span':
            lstm_mse = [0.0019, 0.0023, 0.0032, 0.0033, 0.0045]
            ours_mse = [0.0001, 0.0009, 0.0020, 0.0021, 0.0028]
            mlp_mse = [0.0020, 0.0025, 0.0033, 0.0034, 0.0058]
            tlr_mse = [0.0013, 0.0017, 0.0027, 0.0042, 0.0051]
            lr90_mse = [0.0020, 0.0025, 0.0040, 0.0053, 0.0055]
            lr30_mse = [0.0015, 0.0024, 0.0035, 0.0047, 0.0057]
        else:
            print('name error',name)
            pass
    else:
        print('dataset error')
        pass

    # # # resolution 8i
    #     # LSTM model MSE values
    # lstm_mse = [0.0019, 0.0023, 0.0032, 0.0033, 0.0045]
    
    # # Ours model MSE values
    # ours_mse = [0.0001, 0.0008, 0.0020, 0.0021, 0.0026]
    
    # # MLP model MSE values
    # mlp_mse = [0.002, 0.0025, 0.0039, 0.0034, 0.0058]
    
    # # TLR model MSE values
    # tlr_mse = [0.0013, 0.0017, 0.0027, 0.0042, 0.0051]
    
    # # LR90 model MSE values
    # lr90_mse = [0.002, 0.0025, 0.0040, 0.0053, 0.0055]
    
    # # LR30 model MSE values
    # lr30_mse = [0.0015, 0.0024, 0.0036, 0.0047, 0.0057]




    # Set up the plot
    plt.figure(figsize=(10, 6),dpi=300)
    sns.set_style("whitegrid")  # Use a clean background style
    sns.set_palette("bright")   # Set a bright color palette

    # Plot MSE loss for each model with distinct styles and markers
    # plt.plot(time_ms, mlp_mse, marker='o', linestyle='-', color='blue', label='MLP', linewidth=2)
    plt.plot(time_ms, ours_mse, marker='o', linestyle='-',  label='Ours')
    plt.plot(time_ms, lr30_mse, marker='^', linestyle='--',  label='LR30')
    plt.plot(time_ms, lr90_mse, marker='D', linestyle='--',  label='LR90')
    plt.plot(time_ms, tlr_mse, marker='s', linestyle='--',  label='TLR')
    plt.plot(time_ms, mlp_mse, marker='v', linestyle='--', label='MLP')
    plt.plot(time_ms, lstm_mse, marker='x', linestyle='--', label='LSTM')
    

    # plt.plot(x, T1, label='Ours', linestyle='--', marker='o', markersize=8)
    # plt.plot(x, LR, label='LR', linestyle='--', marker='s', markersize=8)
    # plt.plot(x, TLR, label='TLR', linestyle='--', marker='D', markersize=8)
    # plt.plot(x, MLP, label='M-MLP', linestyle='--', marker='^', markersize=8)
    # plt.plot(x, LSTM, label='LSTM', linestyle='--', marker='x', markersize=8)

    # # Enlarging fonts
    # plt.legend(title="Method", fontsize=16)
    plt.xlabel('Prediction Horizon (ms)', fontsize=18)
    plt.ylabel('MSE Loss', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(time_ms, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('MSE Loss for Cell Visibility Prediction', fontsize=20)

    # Customize the plot
    # plt.xlabel('Time Resolution (ms)', fontsize=18)
    # plt.ylabel('MSE Loss', fontsize=18)
    plt.title(f'{name} prediction on {dataset}', fontsize=16)
    plt.legend(fontsize=16, loc='upper left', frameon=True)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.xscale('log')
    # plt.yscale('log')

    # Set tick marks for x-axis
    # plt.xticks(time_ms, ['33', '333', '1000', '2000', '5000'], fontsize=12)
    # plt.yticks(fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot with high resolution
    # delete '_' in name
    name = name.replace(' ','')
    plt.savefig(f'../result/{dataset}_{name}_mse_loss.png', dpi=300)





if __name__ == '__main__':
    # fsvvd_mse()
    # test_variance()
    # fsvvd_r2()
    dataset = 'FSVVD'
    name_list = ['Viewport Overlap Ratio','Angular Span','Occlusion-aware Visibility','Visibile Angular Span']
    for name in name_list:
        for dataset in ['FSVVD','8i']:
            plot_viewport_fsvvd(dataset,name)
    