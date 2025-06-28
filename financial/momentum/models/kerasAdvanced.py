'''
Class for advanced Keras models with custom layers and training methods.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import numpy as np
import pandas as pd
import keras
from financial.model import KerasModel
import financial.data as fd


'''
class KerasAdvancedModel(KerasModel):

    def __init__(self, name: str, sources: fd.DataDescriptor, target: fd.DataDescriptor, model: keras.Model=None, hyperparameters: dict=None):
        # print(f"Hyperparameters: {hyperparameters}")
        # print(f"{hyperparameters.get('model', {}).get('architecture', 'mlp')} architecture selected for model {name}")
        self.architecture = hyperparameters.get('architecture', 'mlp')
        super().__init__(name, sources, target, model, hyperparameters)
        


    def reshape_input(self, X):
        """
        Reshape input based on architecture. RNN and CNN need 3D input.
        """
        if self.architecture in ["rnn", "lstm", "cnn"]:
            if isinstance(X, pd.DataFrame):
                X = X.values
            n_samples, n_features_total = X.shape
            timesteps = self.hyperparameters["input"]["horizon"]
            assert n_features_total % timesteps == 0, "Incompatible shape: total features no divisible by timesteps"
            n_features = n_features_total // timesteps
            return X.reshape((n_samples, timesteps, n_features))
        return X  # MLP no requiere reshape

    def fit(self, X_train, y_train):

        if self.architecture == "mlp":
            return super().fit(X_train, y_train)

        X_train = self.reshape_input(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

        self.model.compile(
            loss=optimizer_params["loss"],
            optimizer=optimizer_params["optimizer"],
            metrics=optimizer_params["metrics"]
        )

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )

    def predict(self, X):
        if self.architecture == "mlp":
            return super().predict(X)

        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)
    
    def initialize_model(self):
        
        if self.architecture == "mlp":
            return super().initialize_model()

        layers = self.hyperparameters["topology"]["layers"]
        activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()  # Para un ticker sería 1

        model = keras.models.Sequential()

        if self.architecture in ["rnn", "lstm"]:
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            RNNLayer = keras.layers.LSTM if self.architecture == "lstm" else keras.layers.SimpleRNN
            for units in layers[:-1]:
                model.add(RNNLayer(units, return_sequences=True)) # podría ponerse el hidden también
            model.add(RNNLayer(layers[-1]))
            model.add(keras.layers.Dense(1, activation=activation_output))

        elif self.architecture == "cnn":
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            for units in layers[:-1]:
                model.add(keras.layers.Conv1D(filters=units, kernel_size=3, activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(layers[-1], activation=activation_output))

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        return model
'''
    
class RecurrentModel(KerasModel):
    """
    Keras model that supports advanced architectures like RNN LSTM.
    """

    def __init__(self, name: str, sources: fd.DataDescriptor, target: fd.DataDescriptor, model: keras.Model=None, hyperparameters: dict=None):
        self.architecture = hyperparameters.get("model", {}).get("architecture", "lstm")
        super().__init__(name, sources, target, model, hyperparameters)
        
    def reshape_input(self, X):
        """
        Reshape input based on architecture. RNN and CNN need 3D input.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples, n_features_total = X.shape
        timesteps = self.hyperparameters["input"]["horizon"]
        assert n_features_total % timesteps == 0, "Incompatible shape: total features no divisible by timesteps"
        n_features = n_features_total // timesteps
        return X.reshape((n_samples, timesteps, n_features))


    def fit(self, X_train, y_train):

        X_train = self.reshape_input(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

        self.model.compile(
            loss=optimizer_params["loss"],
            optimizer=optimizer_params["optimizer"],
            metrics=optimizer_params["metrics"]
        )

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )

    def predict(self, X):

        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)

    def initialize_model(self):

        layers = self.hyperparameters["topology"]["layers"]
        # activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()  # Para un ticker sería 1

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(horizon, n_features)))
        RNNLayer = keras.layers.LSTM if self.architecture == "lstm" else keras.layers.SimpleRNN
        for units in layers[:-1]:
            model.add(RNNLayer(units, return_sequences=True)) # podría ponerse el hidden también
        model.add(RNNLayer(layers[-1]))
        model.add(keras.layers.Dense(1, activation=activation_output))

        return model
        


class ConvolutionalModel(KerasModel):
    """
    Keras model that supports advanced architectures like CNN.
    """

    def __init__(self, name: str, sources: fd.DataDescriptor, target: fd.DataDescriptor, model: keras.Model=None, hyperparameters: dict=None):
        self.architecture = hyperparameters.get("model", {}).get("architecture", "cnn")
        super().__init__(name, sources, target, model, hyperparameters)

    def reshape_input(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples, n_features_total = X.shape
        timesteps = self.hyperparameters["input"]["horizon"]
        n_features = n_features_total // timesteps

        if self.architecture == "cnn":
            return X.reshape((n_samples, timesteps, n_features))
        elif self.architecture == "cnn2d":
            # reshape to (n_samples, height, width, channels)
            return X.reshape((n_samples, timesteps, n_features, 1))
        else:
            raise ValueError(f"Unsupported architecture for reshape: {self.architecture}")

    def fit(self, X_train, y_train):

        X_train = self.reshape_input(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

        self.model.compile(
            loss=optimizer_params["loss"],
            optimizer=optimizer_params["optimizer"],
            metrics=optimizer_params["metrics"]
        )

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )

    def predict(self, X):

        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)

    def initialize_model(self):
        layers = self.hyperparameters["topology"]["layers"]
        activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()

        model = keras.models.Sequential()

        if self.architecture == "cnn":
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            for units in layers[:-1]:
                model.add(keras.layers.Conv1D(filters=units, kernel_size=3, activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(layers[-1], activation=activation_output))

        elif self.architecture == "cnn2d":
            model.add(keras.layers.Input(shape=(horizon, n_features, 1)))
            for units in layers[:-1]:
                model.add(keras.layers.Conv2D(filters=units, kernel_size=(3, 3), activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(layers[-1], activation=activation_output))

        else:
            raise ValueError(f"Unsupported CNN architecture: {self.architecture}")
    
        return model


from financial.lab.models import ModelFactory
import financial.data as fd

class KerasAdvancedModelFactory(ModelFactory):
    """
    Crea modelos Keras avanzados (RNN, LSTM, etc.) integrados con el sistema de predicción.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor):
        self.architecture = hyperparameters.get("model", {}).get("architecture", {})
        if self.architecture in ["rnn", "lstm"]:
            return RecurrentModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        elif self.architecture in ["cnn", "cnn2d"]:
            return ConvolutionalModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        else:
            return KerasModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)






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

        










        ________________
        ________________

import numpy as np
import pandas as pd
from tensorflow import keras
from financial.model import KerasModel


class KerasAdvancedModel(KerasModel):
    """
    Keras model that supports advanced architectures like RNN, LSTM, and CNN.
    """

    def __init__(self, name, sources, target, model=None, hyperparameters=None):
        super().__init__(name, sources, target, model, hyperparameters)
        self.architecture = hyperparameters.get("model", {}).get("architecture", "mlp")

    def reshape_input(self, X):
        """
        Reshape input based on architecture. RNN and CNN need 3D input.
        """
        if self.architecture in ["rnn", "lstm", "cnn"]:
            # X: DataFrame or 2D array (n_samples, n_features * timesteps)
            if isinstance(X, pd.DataFrame):
                X = X.values
            n_samples, n_features_total = X.shape
            # Horizon es el número de timesteps
            timesteps = self.hyperparameters["input"]["horizon"]
            n_features = n_features_total // timesteps
            return X.reshape((n_samples, timesteps, n_features))
        return X  # MLP expects 2D

    def fit(self, X_train, y_train):
        X_train = self.reshape_input(X_train)

        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

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

    def predict(self, X):
        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)

    def initialize_model(self):
        """
        Construye la red según la arquitectura indicada.
        """
        architecture = self.architecture
        layers = self.hyperparameters["topology"]["layers"]
        activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        input_dim = self.sources.size()
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = 1  # por ahora solo un ticker

        model = keras.models.Sequential()

        if architecture == "mlp":
            model.add(keras.layers.Input(shape=(input_dim,)))
            for units in layers[:-1]:
                model.add(keras.layers.Dense(units, activation=activation_hidden))
            model.add(keras.layers.Dense(layers[-1], activation=activation_output))

        elif architecture in ["rnn", "lstm"]:
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            RNNLayer = keras.layers.LSTM if architecture == "lstm" else keras.layers.SimpleRNN
            for units in layers[:-1]:
                model.add(RNNLayer(units, return_sequences=True))
            model.add(RNNLayer(layers[-1]))
            model.add(keras.layers.Dense(1, activation=activation_output))

        elif architecture == "cnn":
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            for units in layers[:-1]:
                model.add(keras.layers.Conv1D(filters=units, kernel_size=3, activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(layers[-1], activation=activation_output))

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return model

'''