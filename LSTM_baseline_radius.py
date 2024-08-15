import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Function to build an LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100))
    model.add(Dense(9))  # 3 for X, Y, Z and 6 for sin_cos (yaw, pitch, roll)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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
def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Function to save the model
def save_model(model, model_file):
    model.save(model_file)

# Function to load the model
def load_model(model_file):
    model = tf.keras.models.load_model(model_file)
    return model

# Main function
def main():
    window_size = 90
    future_steps = 60
    train_data = read_train_data(['H1', 'H2', 'H3'])
    test_data = read_test_data('H4')
    validation_data = read_validation_data('H4')

    X_train, y_train = get_train_test_data(train_data, window_size=window_size, future_steps=future_steps)
    X_val, y_val = get_train_test_data(validation_data, window_size=window_size, future_steps=future_steps)
    X_test, y_test = get_train_test_data(test_data, window_size=window_size, future_steps=future_steps)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    
    y_pred = model.predict(X_test)
    y_pred_transformed = np.apply_along_axis(convert_back_to_angles, 1, y_pred)

    test_data_pred = test_data.copy()
    test_data_pred.iloc[window_size + future_steps - 1:, 1:7] = y_pred_transformed

    pred_file_path = "../point_cloud_data/LSTM_pred/"
    pred_file_name = f"H4_nav_LSTM_pred{window_size}{future_steps}.csv"
    test_data_pred.to_csv(pred_file_path + pred_file_name, index=False)

if __name__ == '__main__':
    main()
