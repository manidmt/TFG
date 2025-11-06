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

        layers_cfg = self.hyperparameters["topology"]["layers"]
        act_hidden = self.hyperparameters["topology"]["activation"].get("hidden", "relu")
        act_out    = self.hyperparameters["topology"]["activation"]["output"]

        horizon    = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size() // horizon

        mcfg = self.hyperparameters.get("model", {})
        rnn_dropout          = float(mcfg.get("dropout", 0.0))
        rnn_recurrent_drop   = float(mcfg.get("recurrent_dropout", 0.0))
        batch_norm           = bool(mcfg.get("batch_norm", False))
        bidirectional        = bool(mcfg.get("bidirectional", False))
        layer_dropout        = float(mcfg.get("layer_dropout", 0.0))
        dense_head           = list(mcfg.get("dense_head", []))

        RNNLayer = keras.layers.LSTM if self.architecture == "lstm" else keras.layers.SimpleRNN

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(horizon, n_features)))

        for i, units in enumerate(layers_cfg):
            return_seq = (i < len(layers_cfg) - 1)
            rnn = RNNLayer(
                units,
                return_sequences=return_seq,
                dropout=rnn_dropout,
                recurrent_dropout=rnn_recurrent_drop
            )
            if bidirectional:
                rnn = keras.layers.Bidirectional(rnn)

            model.add(rnn)

            if batch_norm:
                model.add(keras.layers.BatchNormalization())

            if layer_dropout > 0.0:
                model.add(keras.layers.Dropout(layer_dropout))

        for units in dense_head:
            model.add(keras.layers.Dense(units, activation=act_hidden))
            if batch_norm:
                model.add(keras.layers.BatchNormalization())
            if layer_dropout > 0.0:
                model.add(keras.layers.Dropout(layer_dropout))

        model.add(keras.layers.Dense(1, activation=act_out))
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
        #print(f"Reshaping input to: (n_samples={n_samples}, timesteps={timesteps}, n_features={n_features})")
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
        n_features = self.sources.size() // horizon # For each ticker we have horizon values, we need to divide by horizon to get the number of tickers

        if self.architecture != "cnn":
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
        hp = self.hyperparameters.get("model", {})
        n_blocks     = hp.get("n_blocks", 2)
        filters      = hp.get("filters", [64, 64])             
        kernel_sizes = hp.get("kernel_sizes", [5, 3])           
        dilations    = hp.get("dilations", 1)                  
        padding      = hp.get("padding", "same")                
        pool_every   = hp.get("pool_every", 1)                  
        pool_size    = hp.get("pool_size", 2)
        dropout      = hp.get("dropout", 0.0)
        l2_reg       = hp.get("l2", 0.0)
        use_bn       = hp.get("batch_norm", True)
        global_pool  = hp.get("global_pool", True)

        # normaliza a listas
        def as_list(v, length):
            if isinstance(v, (list, tuple)):
                return list(v) if len(v) == length else [v]*length
            return [v]*length

        filters      = as_list(filters, n_blocks)
        kernel_sizes = as_list(kernel_sizes, n_blocks)
        dilations    = as_list(dilations, n_blocks)

        reg = keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None

        x = keras.layers.Input(shape=(horizon, n_features))
        y = x

        for b in range(n_blocks):
            y = keras.layers.Conv1D(
                filters=filters[b],
                kernel_size=kernel_sizes[b],
                padding=padding,           
                activation=None,          
                dilation_rate=dilations[b],
                kernel_regularizer=reg,
            )(y)
            if use_bn:
                y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation(activation_hidden)(y)
            if isinstance(pool_every, int) and pool_every > 0 and (b + 1) % pool_every == 0:
                ps = int(pool_size) if (isinstance(pool_size, int) and pool_size > 1) else 2
                y = keras.layers.MaxPooling1D(pool_size=ps)(y)

            if dropout and dropout > 0:
                y = keras.layers.Dropout(dropout)(y)

        if global_pool:
            y = keras.layers.GlobalAveragePooling1D()(y)
        else:
            y = keras.layers.Flatten()(y)

        for units in layers:
            y = keras.layers.Dense(units, activation=activation_hidden, kernel_regularizer=reg)(y)
            if dropout and dropout > 0:
                y = keras.layers.Dropout(dropout)(y)

        out = keras.layers.Dense(1, activation=activation_output)(y)
        model = keras.Model(inputs=x, outputs=out)
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

        horizon    = self.hyperparameters["input"]["horizon"]
        n_features = self.sources.size() // horizon

        act_hidden = self.hyperparameters["topology"]["activation"].get("hidden", "relu")
        act_out    = self.hyperparameters["topology"]["activation"]["output"]

        mcfg = self.hyperparameters.get("model", {})
        n_blocks   = int(mcfg.get("n_blocks", 1))
        num_heads  = int(mcfg.get("num_heads", 2))
        key_dim    = int(mcfg.get("key_dim", 16))
        ff_dim     = int(mcfg.get("ff_dim", 64))
        dropout    = float(mcfg.get("dropout", 0.1))
        causal     = bool(mcfg.get("causal", True))
        global_pool = bool(mcfg.get("global_pool", True))
        dense_head  = list(mcfg.get("dense_head", []))

        inputs = keras.Input(shape=(horizon, n_features))
        x = inputs

        proj_dim = mcfg.get("project_dim", None)
        if proj_dim is not None:
            x = keras.layers.Dense(int(proj_dim))(x)

        for _ in range(n_blocks):
            # Pre-Norm + MultiHeadAttention
            y = keras.layers.LayerNormalization()(x)
            attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
                y, y, use_causal_mask=causal
            )
            y = keras.layers.Dropout(dropout)(attn)
            x = keras.layers.Add()([x, y])

            # Pre-Norm + Feed-Forward
            y = keras.layers.LayerNormalization()(x)
            y = keras.layers.Dense(ff_dim, activation="relu")(y)
            y = keras.layers.Dropout(dropout)(y)
            y = keras.layers.Dense(x.shape[-1])(y)
            x = keras.layers.Add()([x, y])

        # Pooling / Flatten
        if global_pool:
            x = keras.layers.GlobalAveragePooling1D()(x)
        else:
            x = keras.layers.Flatten()(x)

        for units in dense_head:
            x = keras.layers.Dense(units, activation=act_hidden)(x)
            x = keras.layers.Dropout(dropout)(x)

        outputs = keras.layers.Dense(1, activation=act_out)(x)
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
        print(f"[KerasAdvancedModelFactory] Creating model '{model_id}' with architecture '{hyperparameters.get('model', {}).get('architecture', '')}'")
        self.architecture = hyperparameters.get("model", {}).get("architecture", {})
        if self.architecture in ["rnn", "lstm"]:
            return RecurrentModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        elif self.architecture in ["cnn", "cnn2d"]:
            return ConvolutionalModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        elif self.architecture == "transformer":
            return TransformerModel(model_id, input_descriptor, output_descriptor, model=None, hyperparameters=hyperparameters)
        else:
            print(f"[ERROR] Unknown architecture for KerasAdvancedModelFactory: {self.architecture}")
            return None
