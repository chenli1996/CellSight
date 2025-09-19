from ast import main
from re import X
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import random
# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random generator
    torch.cuda.manual_seed(seed)  # PyTorch GPU random generator
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    
    # For deterministic behavior, set the following flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed at the beginning of your script
set_seed(42)

def latest_monotonic_sequence(sequence):
    """ Finds the latest monotonically increasing or decreasing subsequence in a 1D array """
    n = len(sequence)
    if n == 0:
        return np.array([]), np.array([])
    
    # Start from the last element and look backwards
    last_idx = n - 1
    indices = [last_idx]
    values = [sequence[last_idx]]
    
    # Determine if the sequence is increasing or decreasing
    is_increasing = sequence[last_idx - 1] < sequence[last_idx] if last_idx > 0 else True
    
    for i in range(last_idx - 1, -1, -1):
        if (is_increasing and sequence[i] <= sequence[i + 1]) or \
           (not is_increasing and sequence[i] >= sequence[i + 1]):
            indices.append(i)
            values.append(sequence[i])
        else:
            break
    
    # Reverse to make the sequence and indices in ascending order
    indices.reverse()
    values.reverse()
    
    return np.array(indices), np.array(values)


def truncate_linear_regression(history_sequence, future_steps):
    # Extract the latest monotonically increasing subsequence
    indices, values = latest_monotonic_sequence(history_sequence)

    # Reshape indices for sklearn
    indices = np.array(indices).reshape(-1, 1)

    # Create and train the model
    model = LinearRegression()
    model.fit(indices, values)

    # Number of future steps to predict
    # future_steps = 3
    # Create future indices array from the last index of monotonically increasing sequence
    future_indices = np.arange(indices[-1, 0] + 1, indices[-1, 0] + 1 + future_steps).reshape(-1, 1)

    # Predict future values
    future_values = model.predict(future_indices)
    # print(f"Future values for the next {future_steps} steps: {future_values}")
    return future_values


def predict_next_state_tlp(user_data, window_size=2,dof=6,future_steps = 1):
    """
    Predicts the next state based on the last 'window_size' states using linear regression.
    
    Args:
    - user_data: numpy array of shape (n, 6), where n is the number of timesteps,
                 and 6 represents the 6 DoF (x, y, z, yaw, pitch, roll).
    - window_size: int, the number of states to consider for the prediction
    
    Returns:
    - next_state: numpy array of shape (6,), representing the predicted next state.
    """
    if user_data.shape[0] < window_size:
        raise ValueError("Not enough data for prediction.")
    
    next_state = np.zeros(dof)
    time_steps = np.arange(window_size)
    
    # Perform linear regression on each DoF using the last 'window_size' states
    for i in range(dof):
        # m, c = linear_regression(time_steps, user_data[-window_size:, i])
        # next_state[i] = m * (window_size+future_steps-1) + c  # Predict the next state
        future_values = truncate_linear_regression(user_data[-window_size:, i], future_steps)
        next_state[i] = future_values[-1]
    
    return next_state

def linear_regression(x, y):
    """
    Computes the coefficients of a linear regression y = mx + c using least squares.
    
    Args:
    - x: numpy array of shape (n,), the independent variable
    - y: numpy array of shape (n,), the dependent variable
    
    Returns:
    - m: Slope of the fitted line
    - c: Intercept of the fitted line
    """
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def predict_next_state_lp(user_data, window_size=2,dof=6,future_steps = 1):
    """
    Predicts the next state based on the last 'window_size' states using linear regression.
    
    Args:
    - user_data: numpy array of shape (n, 6), where n is the number of timesteps,
                 and 6 represents the 6 DoF (x, y, z, yaw, pitch, roll).
    - window_size: int, the number of states to consider for the prediction
    
    Returns:
    - next_state: numpy array of shape (6,), representing the predicted next state.
    """
    if user_data.shape[0] < window_size:
        raise ValueError("Not enough data for prediction.")
    
    next_state = np.zeros(dof)
    time_steps = np.arange(window_size)
    
    # Perform linear regression on each DoF using the last 'window_size' states
    for i in range(dof):
        m, c = linear_regression(time_steps, user_data[-window_size:, i])
        next_state[i] = m * (window_size+future_steps-1) + c  # Predict the next state
    
    return next_state

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
        # import pdb; pdb.set_trace()
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

        window_data_transformed = convert_to_sin_cos(window_data) # [y,p,r] => [siny, cosy, sinp, cosp, sinr, cosr]
        future_data_transformed = convert_to_sin_cos(future_data[np.newaxis, :])
        # import pdb; pdb.set_trace()
        X.append(window_data_transformed)
        y.append(future_data_transformed[0])  # Flatten here by taking the first (and only) entry

        # Get the original index of the future data point
        original_index = indices[i + future_steps -1]
        sample_indices.append(original_index)  # Record the original index corresponding to this sample

    X = np.array(X)
    y = np.array(y)
    return X, y.reshape(y.shape[0], -1), sample_indices  # Return sample indices


# Training, evaluation, and utility functions
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=5):
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
            best_model = copy.deepcopy(model)  # Save the best model
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model = best_model  # Load the best model
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

# Main function
def main_tlp(future_steps):
    window_size = 90
    downsample_factor = 2  # For 30 FPS

    # Define all users and videos
    all_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing', 'Sweep']
    all_users = ['ChenYongting', 'GuoYushan', 'Guozhaonian', 'HKY', 'RenZhichen',
                 'Sunqiran', 'WangYan', 'fupingyu', 'huangrenyi', 'liuxuya', 'sulehan', 'yuchen']

    # Read and combine all data
    combined_df = read_all_data(all_users, all_videos)

    # Define train, validation, and test videos
    train_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing']
    val_videos = ['Sweep']
    test_videos = ['Sweep']

    # Split data by video
    train_data, val_data, test_data = split_data_by_video(combined_df, train_videos, val_videos, test_videos)

    # Further split 'Sweep' video data into validation and test sets by users
    val_users = all_users[6:]
    test_users = all_users[:6]

    val_data = val_data[val_data['User'].isin(val_users)]
    test_data = test_data[test_data['User'].isin(test_users)]

    # Prepare training and validation data
    X_train, y_train,_ = get_train_test_data(train_data, window_size=window_size,
                                           future_steps=future_steps, downsample_factor=downsample_factor)
    # import pdb; pdb.set_trace()
    X_val, y_val,_ = get_train_test_data(val_data, window_size=window_size,
                                       future_steps=future_steps, downsample_factor=downsample_factor)
    
    X_test, y_test, _ = get_train_test_data(test_data, window_size=window_size,
                                             future_steps=future_steps, downsample_factor=downsample_factor)


    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))


    # Process test data per user
    for user in test_users:
        print(f"Processing predictions for user: {user}")

        user_test_data = test_data[test_data['User'] == user].copy().reset_index(drop=True)

        X_test, y_test, sample_indices = get_train_test_data(user_test_data, window_size=window_size,
                                             future_steps=future_steps, downsample_factor=downsample_factor)

        if len(X_test) == 0:
            print(f"No test data for user {user} after applying window_size and future_steps.")
            continue
        # import pdb; pdb.set_trace()

        # X_test shape is (n_samples, window_size, 9)

        # mse, y_pred = evaluate_model(model, test_loader)
        # X_test = X_test.reshape((X_test.shape[0], -1)) # shape is (n_samples, window_size*9)
        y_pred = np.zeros((X_test.shape[0], 9))
        for i in range(X_test.shape[0]):
            y_pred_i = predict_next_state_tlp(X_test[i], window_size=window_size,dof=9,future_steps=future_steps)
            y_pred[i] = y_pred_i
        
        # import pdb; pdb.set_trace()
        # y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"User: {user}, Mean Squared Error: {mse}")
        # import pdb; pdb.set_trace()
        y_pred_transformed = np.apply_along_axis(convert_back_to_angles, 1, y_pred)
        # import pdb; pdb.set_trace()

        # Map predictions back to original data structure
        # Build a DataFrame with the same structure as the original data
        pred_df = user_test_data.copy()
        # import pdb; pdb.set_trace()
        # set all values to 0
        pred_df.loc[:, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = 0

        # Fill in the predicted values at the corresponding indices
        pred_df.loc[sample_indices, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = y_pred_transformed

        # Save predictions
        pred_file_path = f"../point_cloud_data/TLR_pred_fsvvd/{test_videos[0]}/"
        if not os.path.exists(pred_file_path):
            os.makedirs(pred_file_path)

        pred_file_name = f"{user}_{test_videos[0]}_resampled_pred{window_size}{future_steps}.txt"

        pred_df.to_csv(os.path.join(pred_file_path, pred_file_name), sep='\t', index=False)

def main_lr(future_steps,window_size=90):
    downsample_factor = 2  # For 30 FPS

    # Define all users and videos
    all_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing', 'Sweep']
    all_users = ['ChenYongting', 'GuoYushan', 'Guozhaonian', 'HKY', 'RenZhichen',
                 'Sunqiran', 'WangYan', 'fupingyu', 'huangrenyi', 'liuxuya', 'sulehan', 'yuchen']

    # Read and combine all data
    combined_df = read_all_data(all_users, all_videos)

    # Define train, validation, and test videos
    train_videos = ['Chatting', 'Pulling_trolley', 'News_interviewing']
    val_videos = ['Sweep']
    test_videos = ['Sweep']

    # Split data by video
    train_data, val_data, test_data = split_data_by_video(combined_df, train_videos, val_videos, test_videos)

    # Further split 'Sweep' video data into validation and test sets by users
    val_users = all_users[6:]
    test_users = all_users[:6]

    val_data = val_data[val_data['User'].isin(val_users)]
    test_data = test_data[test_data['User'].isin(test_users)]

    # Prepare training and validation data
    X_train, y_train,_ = get_train_test_data(train_data, window_size=window_size,
                                           future_steps=future_steps, downsample_factor=downsample_factor)
    # import pdb; pdb.set_trace()
    X_val, y_val,_ = get_train_test_data(val_data, window_size=window_size,
                                       future_steps=future_steps, downsample_factor=downsample_factor)
    
    X_test, y_test, _ = get_train_test_data(test_data, window_size=window_size,
                                             future_steps=future_steps, downsample_factor=downsample_factor)


    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))


    # Process test data per user
    for user in test_users:
        print(f"Processing predictions for user: {user}")

        user_test_data = test_data[test_data['User'] == user].copy().reset_index(drop=True)

        X_test, y_test, sample_indices = get_train_test_data(user_test_data, window_size=window_size,
                                             future_steps=future_steps, downsample_factor=downsample_factor)

        if len(X_test) == 0:
            print(f"No test data for user {user} after applying window_size and future_steps.")
            continue
        # import pdb; pdb.set_trace()

        # X_test shape is (n_samples, window_size, 9)

        # mse, y_pred = evaluate_model(model, test_loader)
        # X_test = X_test.reshape((X_test.shape[0], -1)) # shape is (n_samples, window_size*9)
        y_pred = np.zeros((X_test.shape[0], 9))
        for i in range(X_test.shape[0]):
            y_pred_i = predict_next_state_lp(X_test[i], window_size=window_size,dof=9,future_steps=future_steps)
            y_pred[i] = y_pred_i
        

        # y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"User: {user}, Mean Squared Error: {mse}")
        # import pdb; pdb.set_trace()
        y_pred_transformed = np.apply_along_axis(convert_back_to_angles, 1, y_pred)
        # import pdb; pdb.set_trace()

        # Map predictions back to original data structure
        # Build a DataFrame with the same structure as the original data
        pred_df = user_test_data.copy()
        # import pdb; pdb.set_trace()
        # set all values to 0
        pred_df.loc[:, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = 0

        # Fill in the predicted values at the corresponding indices
        pred_df.loc[sample_indices, ['HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ']] = y_pred_transformed

        # Save predictions
        pred_file_path = f"../point_cloud_data/LR_pred_fsvvd/{test_videos[0]}/"
        if not os.path.exists(pred_file_path):
            os.makedirs(pred_file_path)

        pred_file_name = f"{user}_{test_videos[0]}_resampled_pred{window_size}{future_steps}.txt"

        pred_df.to_csv(os.path.join(pred_file_path, pred_file_name), sep='\t', index=False)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for window_size in [30, 90]:
        for future_steps in [1, 10, 30, 60, 150]:
            print(f"Future Steps: {future_steps}, Window Size: {window_size}")
            main_lr(future_steps,window_size=window_size)

    for future_steps in [1, 10, 30, 60, 150]:
        print(f"Future Steps: {future_steps}")
        main_tlp(future_steps)            