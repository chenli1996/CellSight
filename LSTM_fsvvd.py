import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import os

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c_0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Cell state
        
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        
        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Function to read individual user-video data
def read_user_video_data(user, video_name):
    file_path = f'../point_cloud_data/processed_FSVVD/Resample_UB/{video_name}/'
    file_name = f'{user}_{video_name}_resampled.txt'
    full_path = os.path.join(file_path, file_name)

    # Adjust the delimiter and header based on your file's format
    df = pd.read_csv(full_path, sep='\s+')
    df['User'] = user
    df['Video'] = video_name
    df['OriginalIndex'] = df.index  # Add original index
    return df

# Function to read and combine all data
def read_all_data(users, videos):
    dfs = []
    for video in videos:
        for user in users:
            df = read_user_video_data(user, video)
            dfs.append(df)
    combined_df = pd.concat(dfs, axis=0)
    return combined_df

# Function to split data by video
def split_data_by_video(df, train_videos, val_videos, test_videos):
    train_df = df[df['Video'].isin(train_videos)]
    val_df = df[df['Video'].isin(val_videos)]
    test_df = df[df['Video'].isin(test_videos)]
    return train_df, val_df, test_df

# Functions for data preprocessing
def convert_to_sin_cos(data):
    sin_cos_data = []
    for i in range(3):  # Assuming yaw, pitch, roll are the last three columns
        rad_data = np.deg2rad(data[:, i+3])
        sin_data = np.sin(rad_data)
        cos_data = np.cos(rad_data)
        sin_cos_data.append(sin_data)
        sin_cos_data.append(cos_data)
    return np.column_stack((data[:, :3], *sin_cos_data))

def convert_back_to_angles(sin_cos_row):
    angles = []
    for i in range(3):  # Iterate over yaw, pitch, roll sin-cos pairs
        sin_data = sin_cos_row[i*2 + 3]
        cos_data = sin_cos_row[i*2 + 4]
        rad_data = np.arctan2(sin_data, cos_data)
        angles.append(np.rad2deg(rad_data))
    return np.concatenate((sin_cos_row[:3], angles))  # Return X, Y, Z + angles

def get_train_test_data(df, window_size=10, future_steps=30, downsample_factor=1):
    # Adjust the columns to match your data
    data_columns = ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']  # Adjust as needed
    data = df[data_columns].values

    # Get the OriginalIndex column as array
    indices = df['OriginalIndex'].values

    # Downsample the data by taking every 'downsample_factor' frame
    data = data[::downsample_factor]
    indices = indices[::downsample_factor]

    X = []
    y = []
    sample_indices = []

    # Adjust window_size and future_steps according to downsampled data
    for i in range(window_size, len(data) - future_steps + 1):
        window_data = data[i-window_size:i, :]
        future_data = data[i+future_steps-1, :]

        window_data_transformed = convert_to_sin_cos(window_data)
        future_data_transformed = convert_to_sin_cos(future_data[np.newaxis, :])

        X.append(window_data_transformed)
        y.append(future_data_transformed[0])  # Flatten here by taking the first (and only) entry

        # Get the original index of the future data point
        original_index = indices[i + future_steps -1]
        sample_indices.append(original_index)  # Record the original index corresponding to this sample

    X = np.array(X)
    y = np.array(y)
    return X, y.reshape(y.shape[0], -1), sample_indices  # Return sample indices

# Training, evaluation, and utility functions remain the same
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

# Main function adjusted to save predictions per user
def main(future_steps):
    window_size = 90
    # future_steps = 60

    # Define all users and videos
    all_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing', 'Sweep']
    all_users = ['ChenYongting', 'GuoYushan', 'Guozhaonian', 'HKY', 'RenZhichen', 
                 'Sunqiran', 'WangYan', 'fupingyu', 'huangrenyi', 'liuxuya', 'sulehan', 'yuchen']

    # Read and combine all data
    combined_df = read_all_data(all_users, all_videos) # df shape: (n_samples, n_features) (n_samples, 17)
    # import pdb; pdb.set_trace()

    # Define train, validation, and test videos
    train_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing']
    val_videos = ['Sweep']  # We'll split 'Sweep' further for validation and test
    test_videos = ['Sweep']

    # Split data by video
    train_data, val_data, test_data = split_data_by_video(combined_df, train_videos, val_videos, test_videos) # here test and val are same

    # Further split 'Sweep' video data into validation and test sets by users
    test_users = all_users[:6]
    val_users = all_users[6:]
    

    val_data = val_data[val_data['User'].isin(val_users)]
    test_data = test_data[test_data['User'].isin(test_users)]

    # Prepare training and validation datasets
    X_train, y_train, _ = get_train_test_data(train_data, window_size=window_size, future_steps=future_steps, downsample_factor=2) # shape (n_samples, window_size, n_features=9)
    X_val, y_val, _ = get_train_test_data(val_data, window_size=window_size, future_steps=future_steps, downsample_factor=2) # shape (n_samples, window_size, n_features=9)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 60
    output_size = y_train.shape[1]
    # import pdb; pdb.set_trace()
    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    model = train_model(model, train_loader, val_loader)
    
    # Process test data per user
    for user in test_users:
        # import pdb; pdb.set_trace()
        print(f"Processing predictions for user: {user}")

        user_test_data = test_data[test_data['User'] == user].copy()

        # Prepare dataset for the user
        X_test, y_test, sample_indices = get_train_test_data(user_test_data, window_size=window_size, future_steps=future_steps, downsample_factor=2) # shape (n_samples, window_size, n_features=9)

        if len(X_test) == 0:
            print(f"No test data for user {user} after applying window_size and future_steps.")
            continue

        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        mse, y_pred = evaluate_model(model, test_loader)
        print(f"User: {user}, Mean Squared Error: {mse}")
        # import pdb; pdb.set_trace()
        y_pred_transformed = np.apply_along_axis(convert_back_to_angles, 1, y_pred) # Convert back to angles shape (n_samples, 6)
        
        # Build a DataFrame with the same structure as the original data
        pred_df = user_test_data.copy()
        # import pdb; pdb.set_trace()
        # set all values to 0
        pred_df.loc[:, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = 0

        # Fill in the predicted values at the corresponding indices
        pred_df.loc[sample_indices, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = y_pred_transformed

        # Reset index if needed
        # pred_df.reset_index(drop=False, inplace=True)

        # Save the DataFrame to file
        pred_file_path = f"../point_cloud_data/LSTM_pred_fsvvd/{test_videos[0]}/"
        if not os.path.exists(pred_file_path):
            os.makedirs(pred_file_path)

        pred_file_name = f"{user}_{test_videos[0]}_resampled_pred{window_size}{future_steps}.txt"

        # Save the DataFrame
        # Use the same delimiter and format as the input files
        # pred_df.to_csv(os.path.join(pred_file_path, pred_file_name), sep='\t', index=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # for future_steps in [1]:
    for future_steps in [1, 10, 30,60,150]:
        print(f"Future Steps: {future_steps}")
        main(future_steps)