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
    model = MLPRegressor(hidden_layer_sizes=(100,100),max_iter=1000,random_state=21,verbose=True,early_stopping=True)
    return model
# read data
def read_train_data(data_index_list):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
#    data_index = "H4"
    # data_index_list = ['H1','H2','H3']
    # import pdb; pdb.set_trace()
    for data_index in data_index_list:
        file_name = f'{data_index}_nav.csv'
        df = pd.read_csv(file_path+file_name)
        if data_index == 'H1':
            df_all = df
        else:
            df_all = pd.concat([df_all,df],axis=0)        
    return df_all
def read_test_data(data_index):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    # data_index = "H4"
    file_name = f'{data_index}_nav.csv'
    df = pd.read_csv(file_path+file_name)
    # get the Participant 1-14 as test data
    participant = ['P'+str(user_i).zfill(2)+'_V1' for user_i in range(1, 15)]
    df = df[df['Participant'].isin(participant)]
    return df
def read_validation_data(data_index):
    file_path = "../point_cloud_data/6DoF-HMD-UserNavigationData-master/NavigationData/"
    # data_index = "H4"
    file_name = f'{data_index}_nav.csv'
    df = pd.read_csv(file_path+file_name)
    # get the Participant 15-27 as validation data
    participant = ['P'+str(user_i).zfill(2)+'_V1' for user_i in range(15, 28)]
    df = df[df['Participant'].isin(participant)]
    return df
# get the training and testing data
def get_train_test_data(df,window_size=10,future_steps=30):
    # get the data
    data = df.iloc[:,1:7].values
    # import pdb; pdb.set_trace()
    # get the training data
    X = []
    y = []
    for i in range(window_size,len(data)-future_steps+1):
        X.append(data[i-window_size:i,:])
        y.append(data[i+future_steps-1,:])
    X = np.array(X)
    y = np.array(y)
    # import pdb; pdb.set_trace()
    # split the data into training and testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    # return X_train, X_test, y_train, y_test
    import pdb; pdb.set_trace()
    return X, y
# train the model
def train_model(model,X_train,y_train):
    # train the model and show loss
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
    window_size = 90
    future_steps = 1
    train_data = read_train_data(['H1','H2'])
    test_data = read_test_data('H4')
    validation_data = read_validation_data('H4')
    # import pdb; pdb.set_trace()
    X_train, y_train = get_train_test_data(train_data,window_size=window_size,future_steps=future_steps)
    X_test, y_test = get_train_test_data(test_data,window_size=window_size,future_steps=future_steps)
    # import pdb; pdb.set_trace()
    X_val, y_val = get_train_test_data(validation_data,window_size=window_size,future_steps=future_steps)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    print(X_train.shape)

    # build the model
    model = build_mlp_model()
    # train the model with validation data and early stopping
    model = train_model(model,X_train,y_train)
    # evaluate the model
    mse = evaluate_model(model,X_test,y_test)
    print(f'Mean Squared Error:{mse}')
    # predit the test set and write to the file in original format, like the ground truth data
    y_pred = model.predict(X_test)
    test_data_pred = test_data.copy()
    test_data_pred.iloc[window_size+future_steps-1:,1:7] = y_pred
    pred_file_path = "../point_cloud_data/MLP_pred/"
    pred_file_name = f"H4_nav_MLP_pred{window_size}{future_steps}.csv"
    # write the predicted data to the file, the file should have same column names as the ground truth data
    # please check the column names in the ground truth data and use the same column names in the predicted data
    test_data_pred.to_csv(pred_file_path+pred_file_name,index=False)


if __name__ == '__main__':
    main()

