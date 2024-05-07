# use scikit-learn to train a simple MLP model to do the sequence prediction, history window size is 10, predict history is 30 steps
# training set is video H1-H3, test set is video H4
# build a simple MLP model to predict the next 30 steps
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import joblib
# build a simple MLP model
def build_mlp_model():
    model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000, alpha=0.0001,
                         solver='adam', verbose=10, random_state=21,tol=0.000000001)
    return model
# read data
def read_train_data(data_index_list):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
#    data_index = "H4"
    # data_index_list = ['H1','H2','H3']
    for data_index in data_index_list:
        file_name = f'{data_index}_nav.csv'
        df = pd.read_csv(file_path+file_name)
        if data_index == 'H1':
            df_all = df
        else:
            df_all = pd.concat([df_all,df],axis=0)        
    return df
def read_test_data(data_index):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    # data_index = "H4"
    file_name = f'{data_index}_nav.csv'
    df = pd.read_csv(file_path+file_name)
    # get the Participant 1-14 as test data
    df = df[df['Participant'].str.contains('P[0-9]+_V1')]
    return df
# get the training and testing data
def get_train_test_data(df,window_size=10,future_steps=30):
    # get the data
    data = df.iloc[:,1:7].values
    # get the training data
    X = []
    y = []
    for i in range(window_size,len(data)-future_steps+1):
        X.append(data[i-window_size:i,:])
        y.append(data[i+future_steps-1,:])
    X = np.array(X)
    y = np.array(y)
    # split the data into training and testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    # return X_train, X_test, y_train, y_test
    return X, y
# train the model
def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model
# evaluate the model
def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    return mse
# save the model
def save_model(model,model_file):
    joblib.dump(model,model_file)
# load the model
def load_model(model_file):
    model = joblib.load(model_file)
    return model
# main function
def main():
    # read data
    # file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    # data_index = "H4"
    # df = read_data(file_path,data_index)
    # get the training and testing data
    window_size = 10
    future_steps = 30
    train_data = read_train_data(['H1','H2','H3'])
    test_data = read_test_data('H4')
    X_train, y_train = get_train_test_data(train_data,window_size=window_size,future_steps=future_steps)
    X_test, y_test = get_train_test_data(test_data,window_size=window_size,future_steps=future_steps)
    # build the model
    model = build_mlp_model()
    # train the model
    model = train_model(model,X_train,y_train)
    # evaluate the model
    mse = evaluate_model(model,X_test,y_test)
    print(f'Mean Squared Error:{mse}')
    # save the model
    model_file = 'mlp_model.sav'
    save_model(model,model_file)
    # load the model
    model = load_model(model_file)
    # evaluate the model
    mse = evaluate_model(model,X_test,y_test)
    print(f'Mean Squared Error:{mse}')

