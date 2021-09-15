# import required packages
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress tensorflow warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

np.random.seed(39)
tf.get_logger().setLevel(logging.ERROR)  #to suppress tensorflow warnings


if __name__ == "__main__":
    # 1. Load your saved model
    model = keras.models.load_model('models/group_39_RNN_model.h5')

    # 2.a. Load your testing data
    test_df = pd.read_csv("data/test_data_RNN.csv")
    features = ['O1', 'H1', 'L1', 'V1', 'O2', 'H2', 'L2', 'V2', 'O3', 'H3', 'L3', 'V3']
    #print(test_df.head())

    # 2.b. Scaling the data
    scaler = MinMaxScaler()
    test_df[features] = scaler.fit_transform(test_df[features])
    #print(test_df.describe())
    X_test = test_df[features].copy().to_numpy().reshape((-1,12,1))
    y_test = test_df['Open'].copy().to_numpy()

    # 3. Run prediction on the test data and output required plot and loss
    y_pred = model.predict(X_test) 
    print("Testing loss: ", model.evaluate(X_test, y_test)[0])

    sorted_ypred = []
    for i in np.sort(y_test):
        sorted_ypred.append(y_pred[np.where(y_test==i)][0][0])    

    fig, (a1,a2) = plt.subplots(2,1,figsize=(16,20))
    a1.plot(y_test, c='red', label='True values')
    a1.plot(y_pred, c='blue', label='Predicted values')
    a1.legend()
    a1.set_xlabel('Index number')
    a1.set_ylabel('Opening price')

    a2.plot(np.sort(y_test), c='red', label='True values')
    a2.plot(sorted_ypred, c='blue', label='Predicted values')
    a2.legend()
    a2.set_xlabel('Sorted Index number')
    a2.set_ylabel('Opening price')

    plt.show()

