'''
Class for advanced Keras models with custom layers and training methods.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import pandas as pd
import keras
import numpy as np
from financial.model import KerasModel
import financial.data as fd
from financial.lab.models import ModelFactory
import financial.data as fd
from keras.callbacks import LambdaCallback
    
class RecurrentModel(KerasModel):
    """
    Keras model that supports advanced architectures like RNN LSTM.
    It extends the KerasModel class to provide custom training and prediction methods.
    It allows for flexible input reshaping based on the architecture specified in hyperparameters.

    Attributes:
        architecture (str): The architecture of the model (e.g., 'rnn', 'lstm').
        name (str): The name of the model.
        sources (fd.DataDescriptor): The input data descriptor.
        target (fd.DataDescriptor): The target data descriptor.
        model (keras.Model): The Keras model instance.
        hyperparameters (dict): Hyperparameters for the model, including architecture, topology, and input
    """

    def __init__(self, name: str, sources: fd.DataDescriptor, target: fd.DataDescriptor, model: keras.Model=None, hyperparameters: dict=None):
        self.architecture = hyperparameters.get("model", {}).get("architecture", "lstm")
        super().__init__(name, sources, target, model, hyperparameters)
        
    def reshape_input(self, X):
        """
        Reshape input based on architecture. RNN needs 3D input.

        Args:
            X (pd.DataFrame or np.ndarray): Input data to reshape.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples, n_features_total = X.shape
        timesteps = self.hyperparameters["input"]["horizon"]
        assert n_features_total % timesteps == 0, (
            f"Incompatible shape: expected features to be divisible by horizon={timesteps}, "
            f"but got total={n_features_total}"
        )
        n_features = n_features_total // timesteps
        #print(f"Reshaping input to: (n_samples={n_samples}, timesteps={timesteps}, n_features={n_features})")
        return X.reshape((n_samples, timesteps, n_features))

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training target.
        """
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("X_train or y_train contains NaN values.")
        
        X_train = self.reshape_input(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

        self.model.compile(
            loss=optimizer_params["loss"],
            optimizer=optimizer_params["optimizer"],
            metrics=optimizer_params["metrics"]
        )

        # print("▶️ Entrenando modelo...")

        # print(f"X_train before fit: {X_train.shape}, y_train: {y_train.shape}")
        # print(f"Number of {X_train.shape[0]} samples, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")

        # print("X_train shape:", X_train.shape, "mean:", X_train.mean(), "std:", X_train.std())
        # print("y_train.mean():", y_train.mean())
        # print("y_train.std():", y_train.std())
        # print("y_train.min():", y_train.min())
        # print("y_train.max():", y_train.max())


        debug_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: loss={logs['loss']}, val_loss={logs.get('val_loss')}"))

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )
        

    def predict(self, X):
        """
        Predict the target variable using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)
        

    def initialize_model(self):
        """
        Initialize the Keras model based on the specified architecture and hyperparameters.

        Returns:
            keras.Model: The initialized Keras model.
        """

        layers = self.hyperparameters["topology"]["layers"]
        # activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()  # Number of tickers
        #print(self.sources, n_features)

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(horizon, n_features)))
        RNNLayer = keras.layers.LSTM if self.architecture == "lstm" else keras.layers.SimpleRNN
        for units in layers[:-1]:
            model.add(RNNLayer(units, return_sequences=True))
        model.add(RNNLayer(layers[-1]))
        model.add(keras.layers.Dense(1, activation=activation_output))

        return model
 

 
class ConvolutionalModel(KerasModel):
    """
    Keras model that supports advanced architectures like CNN.
    It extends the KerasModel class to provide custom training and prediction methods.
    It allows for flexible input reshaping based on the architecture specified in hyperparameters.

    Attributes:
        architecture (str): The architecture of the model (e.g., 'cnn', 'cnn2d').
        name (str): The name of the model.
        sources (fd.DataDescriptor): The input data descriptor.
        target (fd.DataDescriptor): The target data descriptor.
        model (keras.Model): The Keras model instance.
        hyperparameters (dict): Hyperparameters for the model, including architecture, topology, and input
    """

    def __init__(self, name: str, sources: fd.DataDescriptor, target: fd.DataDescriptor, model: keras.Model=None, hyperparameters: dict=None):
        self.architecture = hyperparameters.get("model", {}).get("architecture", "cnn")
        super().__init__(name, sources, target, model, hyperparameters)

    def reshape_input(self, X):
        """
        Reshape input based on architecture. CNN needs 3D input.
        Args:
            X (pd.DataFrame or np.ndarray): Input data to reshape.
        """

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
        """
        Fit the model to the training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training input data.
            y_train (pd.Series or np.ndarray): Training target data.
        """

        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("X_train or y_train contains NaN values.")

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
        """
        Make predictions using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Input data for predictions.

        Returns:
            np.ndarray: Model predictions.
        """

        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)

    def initialize_model(self):
        """
        Initialize the Keras model based on the specified architecture and hyperparameters.
        """

        layers = self.hyperparameters["topology"]["layers"]
        activation_hidden = self.hyperparameters["topology"]["activation"]["hidden"]
        activation_output = self.hyperparameters["topology"]["activation"]["output"]
        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()

        model = keras.models.Sequential()

        if self.architecture == "cnn":
            model.add(keras.layers.Input(shape=(horizon, n_features)))
            for units in layers:
                model.add(keras.layers.Conv1D(filters=units, kernel_size=horizon, activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1, activation=activation_output))

        elif self.architecture == "cnn2d":
            model.add(keras.layers.Input(shape=(horizon, n_features, 1)))
            for units in layers:
                model.add(keras.layers.Conv2D(filters=units, kernel_size=(horizon, n_features), activation=activation_hidden, padding='same'))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1, activation=activation_output))

        else:
            raise ValueError(f"Unsupported CNN architecture: {self.architecture}")
    
        return model


class TransformerModel(KerasModel):
    """
    Keras model that supports Transformer architecture.
    It extends the KerasModel class to provide custom training and prediction methods.
    It allows for flexible input reshaping based on the architecture specified in hyperparameters.

    Attributes:
        architecture (str): The architecture of the model (e.g., 'transformer').
        name (str): The name of the model.
        sources (fd.DataDescriptor): The input data descriptor.
        target (fd.DataDescriptor): The target data descriptor.
        model (keras.Model): The Keras model instance.
        hyperparameters (dict): Hyperparameters for the model, including architecture, topology, and input
    """

    def __init__(self, name, sources, target, model=None, hyperparameters=None):
        self.architecture = hyperparameters.get("model", {}).get("architecture", "transformer")
        super().__init__(name, sources, target, model, hyperparameters)

    def reshape_input(self, X):
        """
        Reshape input based on architecture. Transformer needs 3D input.
        Args:
            X (pd.DataFrame or np.ndarray): Input data to reshape.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples, n_features_total = X.shape
        timesteps = self.hyperparameters["input"]["horizon"]
        assert n_features_total % timesteps == 0, "Incompatible shape: total features no divisible by timesteps"
        n_features = n_features_total // timesteps
        return X.reshape((n_samples, timesteps, n_features))
    
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training target.
        """

        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("X_train or y_train contains NaN values.")

        X_train = self.reshape_input(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values

        optimizer_params = self.optimizer_hyperparameters()

        self.model.compile(
            loss=optimizer_params["loss"],
            optimizer=optimizer_params["optimizer"],
            metrics=optimizer_params["metrics"]
        )

        # print(f"X_train before fit: {X_train.shape}, y_train: {y_train.shape}")
        # print(f"Number of {X_train.shape[0]} samples, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")

        self.model.fit(
            X_train, y_train,
            epochs=optimizer_params["epochs"],
            batch_size=optimizer_params["batch_size"],
            validation_split=optimizer_params["validation_split"],
            callbacks=[optimizer_params["stop"]] if optimizer_params["stop"] else None
        )
        

    def predict(self, X):
        """
        Predict the target variable using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        X = self.reshape_input(X)
        prediction = self.model.predict(X)
        return prediction if prediction.ndim > 1 else prediction.reshape(-1, 1)

    def initialize_model(self):
        """
        Initialize the Keras model based on the specified architecture.
        """

        horizon = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size()
        num_heads = self.hyperparameters["model"].get("num_heads", 2)
        ff_dim = self.hyperparameters["model"].get("ff_dim", 64)
        dropout_rate = self.hyperparameters["model"].get("dropout", 0.1)
        activation_output = self.hyperparameters["topology"]["activation"]["output"]

        inputs = keras.Input(shape=(horizon, n_features))
        
        # Self-attention
        x = keras.layers.LayerNormalization()(inputs)
        attn_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
        x = keras.layers.Add()([x, attn_output])
        x = keras.layers.Dropout(dropout_rate)(x)
        
        # Feedforward
        x2 = keras.layers.LayerNormalization()(x)
        x2 = keras.layers.Dense(ff_dim, activation="relu")(x2)
        x2 = keras.layers.Dense(n_features)(x2)
        x = keras.layers.Add()([x, x2])
        x = keras.layers.Dropout(dropout_rate)(x)
        
        # Output
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(1, activation=activation_output)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model




class KerasAdvancedModelFactory(ModelFactory):
    """
    Factory for creating advanced Keras models (RNN, LSTM, CNN, Transformer) integrated with the prediction system.
    It follows the same structure as other model factories of the project.

    Methods:
        create_model_from_descriptors(model_id, hyperparameters, input_descriptor, output_descriptor):
            Creates a KerasAdvancedModel instance from the provided descriptors and hyperparameters.
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
        elif self.architecture == "transformer":
            return TransformerModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        else:
            return KerasModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)


# class KerasAdvancedModelFactory2(ModelFactory):
#     """
#     Crea modelos Keras avanzados (RNN, LSTM, CNN, Transformer) integrados con el sistema de predicción.
#     """

#     def create_model_from_descriptors(self, 
#                                       model_id: str, 
#                                       hyperparameters: dict, 
#                                       input_descriptor: fd.DataDescriptor, 
#                                       output_descriptor: fd.DataDescriptor):
#         return KerasAdvancedModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
