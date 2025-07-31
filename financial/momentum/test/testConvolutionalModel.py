'''
Testing ConvolutionalModel class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

import numpy as np
import keras

import financial.data as fd
from financial.momentum.models.kerasAdvanced import ConvolutionalModel

class TestConvolutionalModel(unittest.TestCase):

    def setUp(self):
        self.horizon = 10
        self.variables = [fd.Variable("X1"), fd.Variable("X2")]
        self.sources = fd.Set("input")
        for var in self.variables:
            self.sources.append(var)
        self.target = fd.Variable("Y")
        self.hyperparameters = {
            "input": {"horizon": self.horizon},
            "topology": {
                "layers": [64, 32],
                "activation": {"hidden": "relu", "output": "linear"}
            },
            "model": {
                "architecture": "cnn"
            },
            "optimization": {
                "optimizer": "adam",
                "loss": "mean_squared_error",
                "metrics": ["mae"],
                "epochs": 2,
                "batch_size": 16,
                "validation_split": 0.1,
                "stop": None
            }
        }
        self.model = ConvolutionalModel("CNNModel", self.sources, self.target, hyperparameters=self.hyperparameters)

    def test_reshape_input_cnn(self):
        X = np.random.rand(50, len(self.variables) * self.horizon)
        reshaped = self.model.reshape_input(X)
        self.assertEqual(reshaped.shape, (50, self.horizon, len(self.variables)))

    def test_reshape_input_cnn2d(self):
        self.model.architecture = "cnn2d"
        X = np.random.rand(50, len(self.variables) * self.horizon)
        reshaped = self.model.reshape_input(X)
        self.assertEqual(reshaped.shape, (50, self.horizon, len(self.variables), 1))

    def test_initialize_model_param_count(self):
        model = self.model.initialize_model()
        weights = model.get_weights()
        total_params = np.sum([np.prod(w.shape) for w in weights])
        self.assertGreater(total_params, 0)

    def test_fit_training_runs(self):
        X = np.random.rand(100, len(self.variables) * self.horizon)
        y = np.random.rand(100, 1)
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertEqual(predictions.shape, (100, 1))

    def test_fit_with_nan_values(self):
        X = np.random.rand(50, len(self.variables) * self.horizon)
        X[0, 0] = np.nan
        y = np.random.rand(50, 1)
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

    def test_model_structure_cnn1d(self):
        self.model.architecture = "cnn"
        model = self.model.initialize_model()
        self.assertEqual(model.input_shape, (None, self.horizon, len(self.variables)))
        self.assertEqual(model.output_shape[-1], 1)

    def test_model_structure_cnn2d(self):
        self.model.architecture = "cnn2d"
        model = self.model.initialize_model()
        self.assertEqual(model.input_shape, (None, self.horizon, len(self.variables), 1))
        self.assertEqual(model.output_shape[-1], 1)

    def test_cnn_hidden_layers_structure(self):
        hyperparams = {
            "model": {"architecture": "cnn"},
            "input": {"horizon": self.horizon},
            "topology": {
                "layers": [64, 32],
                "activation": {"hidden": "relu", "output": "linear"}
            },
            "optimization": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"],
                "epochs": 1,
                "batch_size": 8,
                "validation_split": 0.1
            }
        }

        model = ConvolutionalModel("cnn_test", self.sources, self.target, hyperparameters=hyperparams)
        keras_model = model.model
        conv_layers = [layer for layer in keras_model.layers if isinstance(layer, keras.layers.Conv1D)]

        # Solo se crean capas Conv1D para layers[:-1]
        expected_conv_layers = len(hyperparams["topology"]["layers"]) - 1
        self.assertEqual(len(conv_layers), expected_conv_layers)

        # Verifica los filtros de las capas Conv1D
        for i, layer in enumerate(conv_layers):
            expected_filters = hyperparams["topology"]["layers"][i]
            self.assertEqual(layer.filters, expected_filters)

        # Verifica tamaño del kernel
        for layer in conv_layers:
            self.assertEqual(layer.kernel_size[0], self.horizon)

    def test_cnn2d_hidden_layers_structure(self):
        hyperparams = {
            "model": {"architecture": "cnn2d"},
            "input": {"horizon": self.horizon},
            "topology": {
                "layers": [64, 32],
                "activation": {"hidden": "relu", "output": "linear"}
            },
            "optimization": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"],
                "epochs": 1,
                "batch_size": 8,
                "validation_split": 0.1
            }
        }

        model = ConvolutionalModel("cnn2d_test", self.sources, self.target, hyperparameters=hyperparams)
        keras_model = model.model
        conv2d_layers = [layer for layer in keras_model.layers if isinstance(layer, keras.layers.Conv2D)]

        expected_conv_layers = len(hyperparams["topology"]["layers"]) - 1
        self.assertEqual(len(conv2d_layers), expected_conv_layers)

        for i, layer in enumerate(conv2d_layers):
            expected_filters = hyperparams["topology"]["layers"][i]
            self.assertEqual(layer.filters, expected_filters)

            # Verifica el kernel_size esperado para cnn2d: (horizon, n_features)
            expected_kernel_size = (self.horizon, self.sources.size())
            self.assertEqual(layer.kernel_size, expected_kernel_size)

if __name__ == '__main__':
    unittest.main()