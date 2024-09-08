import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# LSTM Model Definition
# Simple LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Cell state
        
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        
        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Function to read training data
def read_train_data(data_index_list):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    for data_index in data_index_list:
        file_name = f'{data_index}_nav.csv'
        df = pd.read_csv(file_path + file_name)
        if data_index == 'H1':
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=0)
    return df_all

# Function to read test data
def read_test_data(data_index):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    file_name = f'{data_index}_nav.csv'
    df = pd.read_csv(file_path + file_name)
    participant = ['P' + str(user_i).zfill(2) + '_V1' for user_i in range(1, 15)]
    df = df[df['Participant'].isin(participant)]
    return df

# Function to read validation data
def read_validation_data(data_index):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    file_name = f'{data_index}_nav.csv'
    df = pd.read_csv(file_path + file_name)
    participant = ['P' + str(user_i).zfill(2) + '_V1' for user_i in range(15, 28)]
    df = df[df['Participant'].isin(participant)]
    return df

# Function to convert yaw, pitch, and roll to sine and cosine components
def convert_to_sin_cos(data):
    sin_cos_data = []
    for i in range(3):  # yaw, pitch, roll are the last three columns
        rad_data = np.deg2rad(data[:, i+3])
        sin_data = np.sin(rad_data)
        cos_data = np.cos(rad_data)
        sin_cos_data.append(sin_data)
        sin_cos_data.append(cos_data)
    return np.column_stack((data[:, :3], *sin_cos_data))

# Function to convert sine and cosine components back to angles
def convert_back_to_angles(sin_cos_data):
    angles = []
    for i in range(3):  # Iterate over yaw, pitch, roll sin-cos pairs
        sin_data = sin_cos_data[i*2 + 3]
        cos_data = sin_cos_data[i*2 + 4]
        rad_data = np.arctan2(sin_data, cos_data)
        angles.append(np.rad2deg(rad_data))
    return np.concatenate((sin_cos_data[:3], angles))  # Return X, Y, Z + angles

# Function to get the training and testing data
def get_train_test_data(df, window_size=10, future_steps=30):
    data = df.iloc[:, 1:7].values
    X = []
    y = []
    for i in range(window_size, len(data) - future_steps + 1):
        window_data = data[i-window_size:i, :]
        future_data = data[i+future_steps-1, :]

        window_data_transformed = convert_to_sin_cos(window_data)
        future_data_transformed = convert_to_sin_cos(future_data[np.newaxis, :])

        X.append(window_data_transformed)
        y.append(future_data_transformed[0])  # Flatten here by taking the first (and only) entry

    X = np.array(X)
    y = np.array(y)
    return X, y.reshape(y.shape[0], -1)  # Flatten y to be 2D

# Function to train the model
# Function to train the model with early stopping
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
        
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        
        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()  # Save the best model
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model)  # Load the best model
            break

    return model


# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(y_batch.numpy())
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    mse = mean_squared_error(y_true, y_pred)
    return mse, y_pred

# Function to save the model
def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)

# Function to load the model
def load_model(model, model_file, input_size, hidden_size, output_size, num_layers=2):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(model_file))
    return model

# Main function
def main(future_steps):
    window_size = 90
    # future_steps = 60
    train_data = read_train_data(['H1', 'H2', 'H3'])
    test_data = read_test_data('H4')
    validation_data = read_validation_data('H4')

    X_train, y_train = get_train_test_data(train_data, window_size=window_size, future_steps=future_steps)
    X_val, y_val = get_train_test_data(validation_data, window_size=window_size, future_steps=future_steps)
    X_test, y_test = get_train_test_data(test_data, window_size=window_size, future_steps=future_steps)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 60
    output_size = y_train.shape[1]

    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    model = train_model(model, train_loader, val_loader)
    
    mse, y_pred = evaluate_model(model, test_loader)
    print(f'Mean Squared Error: {mse}')
    
    y_pred_transformed = np.apply_along_axis(convert_back_to_angles, 1, y_pred)

    test_data_pred = test_data.copy()
    test_data_pred.iloc[window_size + future_steps - 1:, 1:7] = y_pred_transformed

    pred_file_path = "../point_cloud_data/LSTM_pred/"
    pred_file_name = f"H4_nav_LSTM_pred{window_size}{future_steps}.csv"
    test_data_pred.to_csv(pred_file_path + pred_file_name, index=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for future_steps in [10, 30, 60, 150]:
        print(f"Future Steps: {future_steps}")
        main(future_steps)
