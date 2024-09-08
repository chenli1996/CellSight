import numpy as np
from node_feature_utils import parse_trajectory_data
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

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

def predict_next_state_lp_rad(user_data, window_size=2,dof=6,future_steps = 1):
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
    # for i in range(dof):
    #     m, c = linear_regression(time_steps, user_data[-window_size:, i])
    #     next_state[i] = (m * (window_size+future_steps-1) + c +360)%360  # Predict the next state
    for i in range(dof):
        # Convert the data to radians
        rad_data = np.deg2rad(user_data[-window_size:, i])
        
        # Convert the data to sine and cosine values
        sin_data = np.sin(rad_data)
        cos_data = np.cos(rad_data)
        
        # Apply linear regression to the sine and cosine values
        m_sin, c_sin = linear_regression(time_steps, sin_data)
        m_cos, c_cos = linear_regression(time_steps, cos_data)
        
        # Predict the next state
        next_sin = m_sin * (window_size + future_steps - 1) + c_sin
        next_cos = m_cos * (window_size + future_steps - 1) + c_cos
        
        # Convert the sine and cosine values back to an angle in radians
        next_rad = np.arctan2(next_sin, next_cos)
        
        # Convert the angle to degrees and adjust the range to [0, 360)
        next_state[i] = (np.rad2deg(next_rad) + 360) % 360
    
    return next_state

# Example usage
# Assuming user_data is a numpy array with your 6DoF data for user1
# dof = 2
# user_data = np.random.rand(10, dof)  # Dummy data for demonstration

# next_state = predict_next_state_tlp(user_data, window_size=3,dof=dof)
# print("Predicted next state using the last 30 states:", next_state)


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

def predict_next_state_tlp_rad(user_data, window_size=2,dof=6,future_steps = 1):
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
    # for i in range(dof):
    #     m, c = linear_regression(time_steps, user_data[-window_size:, i])
    #     next_state[i] = (m * (window_size+future_steps-1) + c +360)%360  # Predict the next state
    for i in range(dof):
        # Convert the data to radians
        rad_data = np.deg2rad(user_data[-window_size:, i])
        
        # Convert the data to sine and cosine values
        sin_data = np.sin(rad_data)
        cos_data = np.cos(rad_data)
        
        # # Apply linear regression to the sine and cosine values
        # m_sin, c_sin = linear_regression(time_steps, sin_data)
        # m_cos, c_cos = linear_regression(time_steps, cos_data)
        
        # # Predict the next state
        # next_sin = m_sin * (window_size + future_steps - 1) + c_sin
        # next_cos = m_cos * (window_size + future_steps - 1) + c_cos
        future_sin = truncate_linear_regression(sin_data, future_steps)
        future_cos = truncate_linear_regression(cos_data, future_steps)
        next_sin = future_sin[-1]
        next_cos = future_cos[-1]
        
        # Convert the sine and cosine values back to an angle in radians
        next_rad = np.arctan2(next_sin, next_cos)
        
        # Convert the angle to degrees and adjust the range to [0, 360)
        next_state[i] = (np.rad2deg(next_rad) + 360) % 360
    
    return next_state

def linear_regression_baseline():
    # read ground truth data
    window_size_lr = 90
    # future_steps =150
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    data_index = "H4"
    file_name = f'{data_index}_nav.csv'
    pred_file_path = "../point_cloud_data/LR_pred/"

    # for future_steps in [1,30,40,60,90,150]:
    # for future_steps in [1,10,30,60]:
    for future_steps in [150]:
        pred_file_name = f"{data_index}_nav_pred"+str(window_size_lr)+str(future_steps)+".csv"  
        diff_file_name = f"{data_index}_nav_diff"+str(window_size_lr)+str(future_steps)+".csv"
        gt_file_name = f"{data_index}_nav_gt.csv"
        gt_df_all = []
        df_pred_all = []
        diff_all = []
        for user_index_i in tqdm(range(1,28)):
            user_index = 'P'+str(user_index_i).zfill(2)+'_V1'
            trajectory_positions, trajectory_orientations = parse_trajectory_data(file_path+file_name,user_index=user_index)
            # print(trajectory_positions.shape)
            begin_frame_index = 0
            end_frame_index = trajectory_positions.shape[0]-1
            # end_frame_index = 120
            dof = 3
            predicted_trajectory_positions = np.zeros(trajectory_positions[begin_frame_index:end_frame_index+1,:].shape)
            predicted_trajectory_orientations = np.zeros(trajectory_orientations[begin_frame_index:end_frame_index+1,:].shape)
            for frame_index in range(begin_frame_index+window_size_lr,end_frame_index+1 -future_steps +1):
                future_state = predict_next_state_lp(trajectory_positions[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=future_steps)
                predicted_trajectory_positions[frame_index+future_steps -1] = future_state
                future_state = predict_next_state_lp_rad(trajectory_orientations[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=1)
                predicted_trajectory_orientations[frame_index+future_steps -1] = future_state
            gt_df = pd.read_csv(file_path+file_name)
            gt_df = gt_df[gt_df['Participant'] == user_index]
            # initialize df as gt_df copy and replace with the predicted data, we want to keep the original other columns and format
            df = gt_df.copy()
            df.iloc[0:end_frame_index+1,1:4] = predicted_trajectory_positions
            df.iloc[0:end_frame_index+1,4:7] = predicted_trajectory_orientations
            df_pred = df
            # get the difference between the predicted and ground truth data
            diff = df_pred.iloc[0:end_frame_index+1,1:7] - gt_df.iloc[0:end_frame_index+1,1:7]    
            diff.to_csv(pred_file_path+ diff_file_name, index=False)
            gt_df_all.append(gt_df)
            df_pred_all.append(df_pred)
            diff_all.append(diff)
        # concatenate all the dataframes
        gt_df_all = pd.concat(gt_df_all)
        df_pred_all = pd.concat(df_pred_all)
        diff_all = pd.concat(diff_all)
        # write the concatenated dataframes to a file
        gt_df_all.to_csv(pred_file_path+gt_file_name,index=False)
        df_pred_all.to_csv(pred_file_path+pred_file_name,index=False)
        diff_all.to_csv(pred_file_path+diff_file_name, index=False)

def truncated_linear_regression_baseline():
    # read ground truth data
    window_size_lr = 60
    # future_steps =150
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    data_index = "H4"
    file_name = f'{data_index}_nav.csv'
    pred_file_path = "../point_cloud_data/TLR_pred/"

    # for future_steps in [1,30,60,90,150]:
    # for future_steps in [1,10,30,60]:
    for future_steps in [60]:
        pred_file_name = f"{data_index}_nav_tlpred"+str(window_size_lr)+str(future_steps)+".csv"  
        diff_file_name = f"{data_index}_nav_tldiff"+str(window_size_lr)+str(future_steps)+".csv"
        gt_file_name = f"{data_index}_nav_gt.csv"
        gt_df_all = []
        df_pred_all = []
        diff_all = []
        for user_index_i in tqdm(range(1,28)):
            user_index = 'P'+str(user_index_i).zfill(2)+'_V1'
            trajectory_positions, trajectory_orientations = parse_trajectory_data(file_path+file_name,user_index=user_index)
            # print(trajectory_positions.shape)
            begin_frame_index = 0
            end_frame_index = trajectory_positions.shape[0]-1
            # end_frame_index = 120
            dof = 3
            predicted_trajectory_positions = np.zeros(trajectory_positions[begin_frame_index:end_frame_index+1,:].shape)
            predicted_trajectory_orientations = np.zeros(trajectory_orientations[begin_frame_index:end_frame_index+1,:].shape)
            for frame_index in range(begin_frame_index+window_size_lr,end_frame_index+1 -future_steps +1):
                future_state = predict_next_state_tlp(trajectory_positions[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=future_steps)
                # print(trajectory_positions[frame_index-window_size_lr:frame_index,:])
                # print(future_state)
                # import pdb; pdb.set_trace()
                predicted_trajectory_positions[frame_index+future_steps -1] = future_state
                future_state = predict_next_state_tlp_rad(trajectory_orientations[frame_index-window_size_lr:frame_index,:], window_size=window_size_lr,dof=dof,future_steps=1)
                predicted_trajectory_orientations[frame_index+future_steps -1] = future_state
            gt_df = pd.read_csv(file_path+file_name)
            gt_df = gt_df[gt_df['Participant'] == user_index]
            # initialize df as gt_df copy and replace with the predicted data, we want to keep the original other columns and format
            df = gt_df.copy()
            df.iloc[0:end_frame_index+1,1:4] = predicted_trajectory_positions
            df.iloc[0:end_frame_index+1,4:7] = predicted_trajectory_orientations
            df_pred = df
            # get the difference between the predicted and ground truth data
            diff = df_pred.iloc[0:end_frame_index+1,1:7] - gt_df.iloc[0:end_frame_index+1,1:7]    
            diff.to_csv(pred_file_path+ diff_file_name, index=False)
            gt_df_all.append(gt_df)
            df_pred_all.append(df_pred)
            diff_all.append(diff)
        # concatenate all the dataframes
        gt_df_all = pd.concat(gt_df_all)
        df_pred_all = pd.concat(df_pred_all)
        diff_all = pd.concat(diff_all)
        # write the concatenated dataframes to a file
        gt_df_all.to_csv(pred_file_path+gt_file_name,index=False)
        df_pred_all.to_csv(pred_file_path+pred_file_name,index=False)
        diff_all.to_csv(pred_file_path+diff_file_name, index=False)


if __name__ == '__main__':
    # linear_regression()
    linear_regression_baseline()
    truncated_linear_regression_baseline()
    # Example usage
    # history_sequence = -np.array([3, 5, 7, 8, 10, 2, 3, 4, 5, 6])
    # future_steps = 3
    # future_predictions = truncate_linear_regression(history_sequence, future_steps)
    # print(future_predictions)