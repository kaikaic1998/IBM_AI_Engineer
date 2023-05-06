import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense


concrete_data = pd.read_csv('concrete_data.csv')

# split data into predictors and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()

# save the number of predictors to *n_cols* since we will need this number when building our network
n_cols = predictors_norm.shape[1] # number of predictors

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# train and test the model at the same time using the fit() method. 
# We will leave out 30% of the data for validation and we will train the model for 100 epochs
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)