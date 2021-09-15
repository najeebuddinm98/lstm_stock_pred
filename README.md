# LSTM for Stock Prediction

We aim to design and implement a simple LSTM using TensorFlow to predict the opening price of the given stock based on the opening price, highest price, lowest price and the volume of stocks traded in the past 3 days. The dataset used in from a private source, but the model can be tweaked accordingly for another data as the format in which our data is stored in standardised.  

The `report.md` file in the *report* folder has more details on the exploratory data analysis, data preprocessing, model design, training and testing.

## File descriptions

The repo has the following:
- `/data/` = Contains the stock dataset used, named `q2_dataset.csv`. It is split into training and testing sets, which are then stored as separate `csv` files in the same folder.
- `/models/LSTM_model.h5` = This is the trained model stored by the `train_RNN.py` script.
- `/report/` = Contains the primary report describing the entire prototyping and implementation process, along with necessary images as well as training and testing curves.
- `prototyping.ipynb` = As the name suggests, this notebook was used for experimenting with the data processing steps and the model architecture.
- `train_RNN.py` = Script for training the model with the given data. It also contains commented-out code for data preprocessing and splitting steps that was initially performed in the Jupyter notebook, as well as code for generating training performance curves and saving the final model as a `.h5` file. Can be run directly from the terminal
- `test_RNN.py` = Script for testing the model on the unseen test data and displays plot comparing true values to predicted values. Can be run directly from the terminal.


## Our Results

<p align="center">
  <img src="report/testing plot.png" alt="Test MSE" width="700"/>
</p>