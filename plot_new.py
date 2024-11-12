import matplotlib.pyplot as plt
import numpy as np



# fsvvd
# Data
x = [1,10, 30, 60, 150]
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
plt.xlabel('Prediction Horizon (frames)', fontsize=18)
plt.ylabel('MSE', fontsize=18)
# plt.title('MSE Loss for  Prediction Across Different Horizons', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x, fontsize=14)
plt.yticks(fontsize=14)
# plt.ylim(0, 0.03)

plt.savefig('../result/mse_loss.png')
