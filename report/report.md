# RNN for Regression

Here, we predict the opening price of a stock on a given based on the opening price, highest price, lowest price and volume of stock traded in the last 3 business days. Therefore, we have 12 continuous features and 1 continuous output.
 
## Dataset Creation
For creating the dataset, we used the Pandas library. First, we loaded in the given *q2_dataset.csv* using the `read_csv()` function. It is stored in a dataframe with 7 columns and 1259 samples. We rename the columns to `["Date", "Close", "Volume", "Open", "High", "Low"]` for our convenience.  
Since the *Close* column had values with a dollar sign, we used the following code block to remove it and make it a `float64` data-type column.
```
def dollar(s):
	#returns all except the first character in the passed string
    return float(s[1:]) 

df['Close'] = df['Close'].apply(dollar)
```
The *Date* column had dates in two different formats, mainly differing in the length of the year (mm-dd-yy and mm-dd-yyyy). To ensure consistency, we use the `to_datetime()` which converts it to a datetime column as well convert all values into the standard yyyy-mm-dd format. The following code is used to perform this operation
```
df['Date'] = pd.to_datetime(df['Date'])
```
As a precautionary measure, we check for missing values and outliers in the dataset. Neither is present, indicating that we have well recorded data. We also ensure that the dates are in descending order.  
Since we need the values of the previous 3 days for predicting a given day\'s stock, the last 3 samples of our dataframe cannot be used. Thus, it will 1256 samples, with indexes from 0 to 1255. We create the new dataframe `mod_df` with exactly these indexes and the *Date* & *Open* columns from our original dataset. This creation is shown below.
```
mod_df = df.loc[0:1255,['Date','Open']]
```
Now, we need to perform our main operation of getting all the required features within each samples. To get the values of the previous day, we take the values from 1:1256 indexes (1 greater than initial indexes) from the original dataset and join them to the new dataframe, ensuring that the join by matching index values is prevented by using the `to_numpy()` function. This function converts the values to an array thus removing dataframe index values. The same is repeated by increasing the index ranges. The feature names are chosen as `['O1', 'H1', 'L1', 'V1', 'O2', 'H2', 'L2', 'V2', 'O3', 'H3', 'L3', 'V3']`. The code for it is show below.
```
ft = ['Open', 'High', 'Low', 'Volume']

#previous day values
mod_df[['O1','H1','L1','V1']] = df.loc[1:1256, ft].to_numpy() 

#2 days ago values
mod_df[['O2','H2','L2','V2']] = df.loc[2:1257, ft].to_numpy() 

#3days ago values
mod_df[['O3','H3','L3','V3']] = df.loc[3:1258, ft].to_numpy() 
```
We then randomize the dataset using the [`sample()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) function with argument `frac=1`. This function is meant to randomly sample a dataframe from the given dataframe, where the size of the dataframe to be sampled passed as a fraction of the size of the given dataframe. By passing the fraction as 1, we sample the entire dataframe but with shuffled observations. Also, we remove the *Date* column as it is not feature required for model training.
```
final_df = mod_df.sample(frac=1)
dates = final_df.pop('Date')
```
Using the `train_test_split()` from the Scikit-Learn library, we split our dataframe into the training set and test set in a ratio 7:3, as shown below
```
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(final_df, test_size=0.3, random_state=39)
```
Finally, we save the two dataframes as CSV datasets in the *data* directory.
```
train_df.to_csv('data/train_data_RNN.csv', index=False)
test_df.to_csv('data/test_data_RNN.csv', index=False)
```

## Preprocessing
For both the training and test dataset, we perform scaling on all the features as soon as we read them into a pandas dataframe. This is because the input for all TensorFlow models has to be a numeric value between 0 and 1.  
We use min-max scaling for this purpose. The [`MinMaxScaler` class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) from the Scikit_Learn library is perfect for this.  The code to scale the training data is in **train_RNN.py** while the scaling of the test data was done in **test_RNN.py**. Immediately after scaling, we store the respective data into arrays, one for the features and one for the outputs.
```
# for training data
features = ['O1', 'H1', 'L1', 'V1', 
		'O2', 'H2', 'L2', 'V2', 
		'O3', 'H3', 'L3', 'V3']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])

X_train = train_df[features].copy().to_numpy().reshape((-1,12,1))
y_train = train_df['Open'].copy().to_numpy()
```

```
# for testing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
test_df[features] = scaler.fit_transform(test_df[features])

X_test = test_df[features].copy().to_numpy().reshape((-1,12,1))
y_test = test_df['Open'].copy().to_numpy()
```
Above, we reshape the feature arrays *X_train* and *X_test* because TensorFlow expects the input dimensions to be in the form (number of samples, number of features, number of sequences). Since all the features for a given sample are passed along with it, the number of sequences is kept 1.  

## Design steps
Throughout the entire process, we aimed to gradually increase the complexity of the model; by using advanced layers, more layers, more nodes etc. Once a model archotecture was chosen, we attempted hyperparameter tuning. Here, hyperparameters include type of optimizer, learning rate, batch size  and number of epochs. We kept our loss function as mean absolute error (MAE) and our metric as root mean squared error (RMSE).  
We started with a model consisting of a
* RNN layer - 128 nodes, ReLU activation
* RNN layer - 128 nodes, ReLU activation
* Dense layer - 32 nodes, ReLU activation
* Dense layer - 1 node, Linear activation  
* Batch size = 32, Epoch = 50

We tried 3 optimizers; Stochastic Gradient Descent, RMSProp and Adam. [Adam](https://keras.io/api/optimizers/adam/) gave the lowest loss of 6.2 at the default learning rate of 0.01. Increasing the learning rate did not change the final loss value whereas decreasing it required double the number of training epochs to get the same loss as that of 0.01. When the loss function graph was plotted against epochs, oscillations were observed throughout the curve. To mitigate this, batch size was reduced to 16 and increased the epochs to 100. This reduced the oscillations and gave a more consistent result over a series of runs.  

Next, we chose a model consisting of a
* LSTM layer - 64 nodes, ReLU activation
* LSTM layer - 64 nodes, ReLU activation
* Dense layer - 32 nodes, ReLU activation
* Dense layer - 1 node, Linear activation  
* Batch size = 32, Epoch = 50

Again, Adam optimizer with a learning rate of 0.01 gave the lowest loss of 4.5. Increasing the learning rate increased the oscillations greatly in the loss function curve whereas decreasing it required triple the number of training epochs to get the same loss as that of 0.01. Oscillations were still present in the curve and decreasing the batch size did not improve the situation. By increasing the epoch to 100, we got more consistent results. The final training loss still would oscillate but was always in the range [3,6], much better than the previous model. The reason we chose lesser nodes per recurrent layer here is because LSTM cells have more trainable parameters compared to simple RNN cells.  
To build upon the above model, we added more LSTM and Dense layers but no significant improvement was observed and more the number of layers, greater is the chance of overfitting. At this stage, we trained the model and evaluated it on the test dataset. Our test losses were similar to the training loss meaning that our model was not overfitting our training data, so we chose not to experiment with addition of Dropout layers.  

## Final Architecture & Hyperparameters
The final model has the architecture:
* LSTM layer - 64 nodes, ReLU activation
* LSTM layer - 64 nodes, ReLU activation
* Dense layer - 32 nodes, ReLU activation
* Dense layer - 1 node, Linear activation  

Hyperparameter values:
* Optimizer = Adam
* Learning rate = 0.01
* Loss function = mean absolute error
* Metric = root mean squared error
* Batch size = 32
* Epochs = 100

The code implementing the above is shown as follows

```
model = keras.models.Sequential()

model.add(keras.layers.LSTM(64, activation='relu', return_sequences=True, 
							input_shape=(12,1)))
model.add(keras.layers.LSTM(64, activation='relu'))

model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
#model.summary()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss="mean_absolute_error",
		 metrics=[keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, y_train, batch_size=32, epochs=100)
```

The output of the `summary()` function is shown below

<p>
    <img src="model summary.png" alt="Model summary" width="400" height="220" />
</p>

## Training Output
We run the **train_RNN.py** file in our terminal and the following training loop is obtained

```
If first run, importing packages takes a minute
Epoch 1/100
28/28 [==============================] - 2s 10ms/step - 
loss: 158.5894 - root_mean_squared_error: 181.5784
Epoch 2/100
28/28 [==============================] - 0s 10ms/step - 
loss: 69.3168 - root_mean_squared_error: 92.6765
Epoch 3/100
28/28 [==============================] - 0s 10ms/step - 
loss: 10.6792 - root_mean_squared_error: 13.7535
Epoch 4/100
28/28 [==============================] - 0s 9ms/step - 
loss: 9.8346 - root_mean_squared_error: 12.4946
Epoch 5/100
28/28 [==============================] - 0s 9ms/step - 
loss: 12.9749 - root_mean_squared_error: 16.9034
Epoch 6/100
28/28 [==============================] - 0s 10ms/step - 
loss: 13.2104 - root_mean_squared_error: 17.0900
Epoch 7/100
28/28 [==============================] - 0s 9ms/step - 
loss: 12.1656 - root_mean_squared_error: 15.6269
Epoch 8/100
28/28 [==============================] - 0s 10ms/step - 
loss: 11.4262 - root_mean_squared_error: 14.3654
Epoch 9/100
28/28 [==============================] - 0s 9ms/step - 
loss: 26.2587 - root_mean_squared_error: 31.6918
Epoch 10/100
28/28 [==============================] - 0s 10ms/step - 
loss: 26.0492 - root_mean_squared_error: 35.0969
Epoch 11/100
28/28 [==============================] - 0s 10ms/step - 
loss: 7.2684 - root_mean_squared_error: 10.2999
Epoch 12/100
28/28 [==============================] - 0s 10ms/step - 
loss: 7.8613 - root_mean_squared_error: 10.3261
Epoch 13/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.5081 - root_mean_squared_error: 8.3151
Epoch 14/100
28/28 [==============================] - 0s 10ms/step - 
loss: 15.8499 - root_mean_squared_error: 19.3732
Epoch 15/100
28/28 [==============================] - 0s 11ms/step - 
loss: 16.8123 - root_mean_squared_error: 20.0942
Epoch 16/100
28/28 [==============================] - 0s 11ms/step - 
loss: 14.5930 - root_mean_squared_error: 17.9661
Epoch 17/100
28/28 [==============================] - 0s 11ms/step - 
loss: 12.4486 - root_mean_squared_error: 15.7860
Epoch 18/100
28/28 [==============================] - 0s 10ms/step - 
loss: 23.5457 - root_mean_squared_error: 28.7267
Epoch 19/100
28/28 [==============================] - 0s 10ms/step - 
loss: 20.3574 - root_mean_squared_error: 24.9838
Epoch 20/100
28/28 [==============================] - 0s 12ms/step - 
loss: 6.7952 - root_mean_squared_error: 8.6976
Epoch 21/100
28/28 [==============================] - 0s 11ms/step - 
loss: 6.3193 - root_mean_squared_error: 8.8024
Epoch 22/100
28/28 [==============================] - 0s 9ms/step - 
loss: 6.6355 - root_mean_squared_error: 8.9107
Epoch 23/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.2608 - root_mean_squared_error: 8.6687
Epoch 24/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.8625 - root_mean_squared_error: 6.9336
Epoch 25/100
28/28 [==============================] - 0s 9ms/step - 
loss: 6.6462 - root_mean_squared_error: 8.6839
Epoch 26/100
28/28 [==============================] - 0s 10ms/step - 
loss: 7.1688 - root_mean_squared_error: 9.6278
Epoch 27/100
28/28 [==============================] - 0s 10ms/step - 
loss: 8.1712 - root_mean_squared_error: 10.8435
Epoch 28/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.1089 - root_mean_squared_error: 8.1583
Epoch 29/100
28/28 [==============================] - 0s 11ms/step - 
loss: 9.0739 - root_mean_squared_error: 11.6346
Epoch 30/100
28/28 [==============================] - 0s 11ms/step - 
loss: 5.5374 - root_mean_squared_error: 7.5405
Epoch 31/100
28/28 [==============================] - 0s 11ms/step - 
loss: 6.5100 - root_mean_squared_error: 8.5272
Epoch 32/100
28/28 [==============================] - 0s 11ms/step - 
loss: 7.4538 - root_mean_squared_error: 9.4458
Epoch 33/100
28/28 [==============================] - 0s 10ms/step - 
loss: 7.2294 - root_mean_squared_error: 9.1120
Epoch 34/100
28/28 [==============================] - 0s 11ms/step - 
loss: 7.5004 - root_mean_squared_error: 9.5962
Epoch 35/100
28/28 [==============================] - 0s 9ms/step - 
loss: 7.6638 - root_mean_squared_error: 9.8622
Epoch 36/100
28/28 [==============================] - 0s 9ms/step - 
loss: 7.9221 - root_mean_squared_error: 10.1407
Epoch 37/100
28/28 [==============================] - 0s 11ms/step - 
loss: 8.2313 - root_mean_squared_error: 10.1059
Epoch 38/100
28/28 [==============================] - 0s 11ms/step - 
loss: 8.3677 - root_mean_squared_error: 10.7384
Epoch 39/100
28/28 [==============================] - 0s 12ms/step - 
loss: 5.0437 - root_mean_squared_error: 6.8057
Epoch 40/100
28/28 [==============================] - 0s 11ms/step - 
loss: 6.2823 - root_mean_squared_error: 8.2253
Epoch 41/100
28/28 [==============================] - 0s 11ms/step - 
loss: 4.6452 - root_mean_squared_error: 6.3911
Epoch 42/100
28/28 [==============================] - 0s 11ms/step - 
loss: 5.7657 - root_mean_squared_error: 8.0114
Epoch 43/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.0554 - root_mean_squared_error: 8.2653
Epoch 44/100
28/28 [==============================] - 0s 11ms/step - 
loss: 8.0827 - root_mean_squared_error: 10.1294
Epoch 45/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.2797 - root_mean_squared_error: 6.1476
Epoch 46/100
28/28 [==============================] - 0s 9ms/step - 
loss: 6.1169 - root_mean_squared_error: 8.2304
Epoch 47/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.8996 - root_mean_squared_error: 6.5734
Epoch 48/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.9061 - root_mean_squared_error: 7.9985
Epoch 49/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.1452 - root_mean_squared_error: 7.0773
Epoch 50/100
28/28 [==============================] - 0s 9ms/step - 
loss: 7.3000 - root_mean_squared_error: 9.8779
Epoch 51/100
28/28 [==============================] - 0s 9ms/step - 
loss: 11.1526 - root_mean_squared_error: 14.0677
Epoch 52/100
28/28 [==============================] - 0s 11ms/step - 
loss: 5.1679 - root_mean_squared_error: 7.3004
Epoch 53/100
28/28 [==============================] - 0s 11ms/step - 
loss: 4.5120 - root_mean_squared_error: 6.3250
Epoch 54/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.8355 - root_mean_squared_error: 6.7825
Epoch 55/100
28/28 [==============================] - 0s 10ms/step - 
loss: 8.1690 - root_mean_squared_error: 9.9855
Epoch 56/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.2566 - root_mean_squared_error: 7.0973
Epoch 57/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.6275 - root_mean_squared_error: 7.6955
Epoch 58/100
28/28 [==============================] - 0s 10ms/step - 
loss: 8.1470 - root_mean_squared_error: 10.3489
Epoch 59/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.0049 - root_mean_squared_error: 6.7948
Epoch 60/100
28/28 [==============================] - 0s 9ms/step - 
loss: 6.2235 - root_mean_squared_error: 8.2272
Epoch 61/100
28/28 [==============================] - 0s 10ms/step - 
loss: 9.9689 - root_mean_squared_error: 12.5844
Epoch 62/100
28/28 [==============================] - 0s 11ms/step - 
loss: 7.9574 - root_mean_squared_error: 10.3873
Epoch 63/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.1097 - root_mean_squared_error: 7.1505
Epoch 64/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.0573 - root_mean_squared_error: 5.8295
Epoch 65/100
28/28 [==============================] - 0s 10ms/step - 
loss: 9.5980 - root_mean_squared_error: 12.1570
Epoch 66/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.2834 - root_mean_squared_error: 5.9696
Epoch 67/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.1670 - root_mean_squared_error: 7.7629
Epoch 68/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.2914 - root_mean_squared_error: 7.0576
Epoch 69/100
28/28 [==============================] - 0s 11ms/step - 
loss: 5.0524 - root_mean_squared_error: 6.9001
Epoch 70/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.9762 - root_mean_squared_error: 6.7983
Epoch 71/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.3846 - root_mean_squared_error: 7.2002
Epoch 72/100
28/28 [==============================] - 0s 11ms/step - 
loss: 11.0186 - root_mean_squared_error: 13.2810
Epoch 73/100
28/28 [==============================] - 0s 11ms/step - 
loss: 5.9709 - root_mean_squared_error: 7.6163
Epoch 74/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.2881 - root_mean_squared_error: 6.0743
Epoch 75/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.9681 - root_mean_squared_error: 7.7850
Epoch 76/100
28/28 [==============================] - 0s 9ms/step - 
loss: 8.6823 - root_mean_squared_error: 11.0235
Epoch 77/100
28/28 [==============================] - 0s 9ms/step - 
loss: 6.7663 - root_mean_squared_error: 8.7188
Epoch 78/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.8930 - root_mean_squared_error: 7.8168
Epoch 79/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.6196 - root_mean_squared_error: 6.2482
Epoch 80/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.5524 - root_mean_squared_error: 6.3548
Epoch 81/100
28/28 [==============================] - 0s 9ms/step - 
loss: 7.9745 - root_mean_squared_error: 10.3928
Epoch 82/100
28/28 [==============================] - 0s 10ms/step - 
loss: 9.7659 - root_mean_squared_error: 12.1273
Epoch 83/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.4553 - root_mean_squared_error: 6.1572
Epoch 84/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.6437 - root_mean_squared_error: 7.5020
Epoch 85/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.6372 - root_mean_squared_error: 6.4031
Epoch 86/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.1797 - root_mean_squared_error: 8.2603
Epoch 87/100
28/28 [==============================] - 0s 10ms/step - 
loss: 6.7927 - root_mean_squared_error: 9.1587
Epoch 88/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.9750 - root_mean_squared_error: 6.8095
Epoch 89/100
28/28 [==============================] - 0s 9ms/step - 
loss: 8.3797 - root_mean_squared_error: 10.7708
Epoch 90/100
28/28 [==============================] - 0s 10ms/step - 
loss: 5.3625 - root_mean_squared_error: 7.0099
Epoch 91/100
28/28 [==============================] - 0s 9ms/step - 
loss: 3.9776 - root_mean_squared_error: 5.5874
Epoch 92/100
28/28 [==============================] - 0s 9ms/step - 
loss: 5.3970 - root_mean_squared_error: 7.1365
Epoch 93/100
28/28 [==============================] - 0s 9ms/step - 
loss: 4.6853 - root_mean_squared_error: 6.3591
Epoch 94/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.3266 - root_mean_squared_error: 5.9840
Epoch 95/100
28/28 [==============================] - 0s 9ms/step - 
loss: 3.6295 - root_mean_squared_error: 5.1982
Epoch 96/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.5215 - root_mean_squared_error: 5.8966
Epoch 97/100
28/28 [==============================] - 0s 9ms/step - 
loss: 3.9277 - root_mean_squared_error: 5.2657
Epoch 98/100
28/28 [==============================] - 0s 10ms/step - 
loss: 3.7527 - root_mean_squared_error: 5.1155
Epoch 99/100
28/28 [==============================] - 0s 10ms/step - 
loss: 3.7274 - root_mean_squared_error: 5.2293
Epoch 100/100
28/28 [==============================] - 0s 10ms/step - 
loss: 4.4917 - root_mean_squared_error: 6.3656

 Final Training loss:  4.491659164428711
 ```
 
While our loss value consistently reached less than 9 in just 60 epochs, we still observe a lot of oscillations. These oscillations seem to stabilise after 80 epochs. This is why we used epoch=100.  
Our final training loss is 4.5. This means that on average, our predictions are &plusmn;4.5 dollars near the actual stock price.  
We retrained the model over 12 times and the final loss never exceeded 5.7. These slight variations might be due to the randomised initialisations of weights at the start of the learning phase as well as due to the optimizer finding different local minimum while training.  
We plotted the training loss and training RMSE against epoch values. The following graphs are obtained
<p>
    <img src="training plot.png" alt="Training plot" width="790" height="380" />
</p>
.

## Testing Output
We run **testing_RNN.py** in our terminal and the final result is obtained
```
12/12 [==============================] - 0s 3ms/step - 
loss: 3.9689 - root_mean_squared_error: 5.2760
Testing loss:  3.968895673751831
```
The testing loss is 4, which is very slightly better than our training loss. This indicates that our model is neither overfitting nor underfitting, which is ideal.  
We plot the true and predicted values together to see how they differ. We create two plots here, one with the values plotted against their true indexes while the other has the indexes sorted such that the true values are in ascending order.

<p>
    <img src="testing plot.png" alt="Testing plot" width="790" height="380" />
</p>

As seen above, our predicted values are quite close to the true values. There are a few significant spikes especially between opening prices between 200 and 300 dollars , but there does not seem to a pattern to them and they seem like a few outliers with higher errors than the rest of the curve.

## More Days for Features
We tried using the values from the past 5 days as features (total features=20) to the model. The results were close to identical to that obtained above, but there were fewer oscillations in the training loss curve.  
Then, we tried using the values of previous 10 days as features (total features=40) and using the same model gave use worse results than both the 3-day and 5-day model. We added another LSTM layer and retrained the model with the 10-day features dataset and only then obtained low enough training loss. The training loss curve here also had fewer oscillations than the 3-day one but similar to the 5-day one.  
We haven\'t tried other variations, but we can give a naive conclusion that more the number of features, more complex of a model is needed to get good results. The one advantage observed is more stability in the final training loss, showing lessening of effect of random weight initialisation. The disadvantage is that more complex the model needed, more data as well as computational time is required for training.