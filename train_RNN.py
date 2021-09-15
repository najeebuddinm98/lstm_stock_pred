# import required packages
print('If first run, importing packages takes a minute')
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to suppress tensorflow warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

np.random.seed(39)
tf.get_logger().setLevel(logging.ERROR)  #to suppress tensorflow warnings

### DATASET CREATION
## 
##df = pd.read_csv('data/q2_dataset.csv')
##df.columns = ["Date", "Close", "Volume", "Open", "High", "Low"] #renaming the columns for easier use
##
##def dollar(s):
##    #for removing the first index of the string i.e. the dollar sign
##    return float(s[1:])
##
##df['Close'] = df['Close'].apply(dollar) #remove the dollar sign
##df['Date'] = pd.to_datetime(df['Date']) #convert column to DateTime object so that all values are in yyyy-mm-dd
##
##new_df = df.loc[0:1255,['Date','Open']]
##
###we now save the the required values from previous days under new columns as shown below. 
###The indexing controls the selection of values from the previous days.
##new_df[['O1','H1','L1','V1']] = df.loc[1:1256, ['Open', 'High', 'Low', 'Volume']].to_numpy()
##new_df[['O2','H2','L2','V2']] = df.loc[2:1257, ['Open', 'High', 'Low', 'Volume']].to_numpy()
##new_df[['O3','H3','L3','V3']] = df.loc[3:1258, ['Open', 'High', 'Low', 'Volume']].to_numpy()
## 
##final_df = new_df.sample(frac=1, random_state=39) #this is to randomize the observations
##dates = final_df.pop('Date') #Date value was only required for creating the new dataset but not for the model, so can be removed
##
##train_df, test_df = train_test_split(final_df, test_size=0.3, random_state=39) #splitting the dataset into training and test set
##train_df.to_csv('data/train_data_RNN.csv', index=False)
##test_df.to_csv('data/test_data_RNN.csv', index=False)

if __name__ == "__main__":
    # 1.a. load your training data
    train_df = pd.read_csv("data/train_data_RNN.csv")
    features = ['O1', 'H1', 'L1', 'V1', 'O2', 'H2', 'L2', 'V2', 'O3', 'H3', 'L3', 'V3']
    #print(train_df.describe())

    # 1.b. preprocessing - scaling the features as TensorFlow input has to be between 0 and 1
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    #print(train_df.describe())
    X_train = train_df[features].copy().to_numpy().reshape((-1,12,1))
    y_train = train_df['Open'].copy().to_numpy()

    # 2. Train your network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, activation='relu', return_sequences = True, input_shape = (12,1)))
    model.add(keras.layers.LSTM(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    #model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="mean_absolute_error", metrics=[keras.metrics.RootMeanSquaredError()])
    history = model.fit(X_train, y_train, batch_size=32, epochs=100) #will print loss at each step
    print("\n Final Training loss: ", history.history['loss'][-1]) #final training loss

    #generating loss and RMSE curves on training data for report
    #fig, (a1,a2) = plt.subplots(1,2,figsize=(30,7))
    #a1.plot(history.history['loss'], c='red')
    #a1.set_xlabel('Epochs')
    #a1.set_ylabel('Training Loss (Mean Absolute Error)')
    #a2.plot(history.history['root_mean_squared_error'], c='blue')
    #a2.set_xlabel('Epochs')
    #a2.set_ylabel('Training RMSE')
    #plt.show()
    
    # 3. Save your model
    keras.models.save_model(model, 'models/LSTM_model.h5')
