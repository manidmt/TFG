'''
Class for advanced Keras models with custom layers and training methods.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import numpy as np
import keras

class KerasAdvancedModel:

    def __init__(self, datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead=20, horizon=90):
        self.datastore = datastore
        self.ticker = ticker
        self.model_factory = model_factory
        self.name = name
        self.start_year = start_year
        self.end_year = end_year
        self.hyperparameters = hyperparameters
        self.lookahead = lookahead
        self.horizon = horizon
        self.predictions = None

    def get_training_data(self):
        data = self.datastore.get_data(self.ticker, self.start_year, self.end_year)
        y = data[self.ticker].iloc[self.horizon:]
        samples = len(y)
        features = len(data.columns)
        X = np.zeros((samples, self.horizon, features))
        for index in range(samples):
            X[index] = data.iloc[index:index + self.horizon, :]
        return X, y


    def keras(timesteps: int, features: int, cells: int, type) -> keras.Model:
        inputs = keras.layers.Input(shape=(timesteps, features))
        if type == 'LSTM':
            lstm_out = keras.layers.LSTM(cells)(inputs)
        elif type == 'GRU':
            lstm_out = keras.layers.GRU(cells)(inputs)
        else:
            lstm_out = keras.layers.SimpleRNN(cells)(inputs)

        outputs = keras.layers.Dense(1, activation='linear')(lstm_out)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def fit(self, X_train, y_train):
        

        self.model.compile(
            optimizer=optimizer_params["optimizer"],
            loss=optimizer_params["loss"],
            metrics=optimizer_params["metrics"]
        )

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )