'''
Testing TransformerModel class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

import numpy as np
import keras

import financial.data as fd
from financial.momentum.models.kerasAdvanced import TransformerModel

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        self.horizon = 10
        self.variables = [fd.Variable("X1"), fd.Variable("X2")]
        self.sources = fd.Set("input")
        for var in self.variables:
            self.sources.append(var)
        self.target = fd.Variable("Y")

        self.hyperparams = {
            "model": {
                "architecture": "transformer",
                "num_heads": 2,
                "ff_dim": 32,
                "dropout": 0.1
            },
            "input": {"horizon": self.horizon},
            "topology": {
                "activation": {"output": "linear"}
            },
            "optimization": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"],
                "epochs": 2,
                "batch_size": 16,
                "validation_split": 0.1
            }
        }

        self.model = TransformerModel("transformer_test", self.sources, self.target, hyperparameters=self.hyperparams)

    def test_input_reshape(self):
        X = np.random.rand(100, len(self.variables) * self.horizon)
        reshaped = self.model.reshape_input(X)
        self.assertEqual(reshaped.shape, (100, self.horizon, len(self.variables)))

    def test_output_shape(self):
        model = self.model.model
        self.assertEqual(model.output_shape, (None, 1))

    def test_model_trains_and_predicts(self):
        X = np.random.rand(100, len(self.variables) * self.horizon)
        y = np.random.rand(100, 1)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.assertEqual(preds.shape, (100, 1))

    def test_attention_layer_presence(self):
        attention_layers = [layer for layer in self.model.model.layers if isinstance(layer, keras.layers.MultiHeadAttention)]
        self.assertEqual(len(attention_layers), 1)

    def test_layer_normalization_count(self):
        norm_layers = [layer for layer in self.model.model.layers if isinstance(layer, keras.layers.LayerNormalization)]
        self.assertEqual(len(norm_layers), 2)  # Before attention and before FF

    def test_dropout_layers_count(self):
        dropout_layers = [layer for layer in self.model.model.layers if isinstance(layer, keras.layers.Dropout)]
        self.assertEqual(len(dropout_layers), 2)  # After attention and after FF

    def test_feedforward_structure(self):
        dense_layers = [layer for layer in self.model.model.layers if isinstance(layer, keras.layers.Dense)]
        ff_dim = self.hyperparams["model"]["ff_dim"]
        output_activation = self.hyperparams["topology"]["activation"]["output"]
        # Esperamos: 1 FF (relu), 1 FF (proyección), 1 output
        self.assertGreaterEqual(len(dense_layers), 3)
        self.assertEqual(dense_layers[-1].units, 1)
        self.assertEqual(dense_layers[-1].activation.__name__, output_activation)

    def test_model_has_global_average_pooling(self):
        pooling_layers = [layer for layer in self.model.model.layers if isinstance(layer, keras.layers.GlobalAveragePooling1D)]
        self.assertEqual(len(pooling_layers), 1)

    def test_model_handles_varied_heads_ffdim(self):
        # Cambio de heads y ff_dim
        self.hyperparams["model"]["num_heads"] = 4
        self.hyperparams["model"]["ff_dim"] = 128
        model = TransformerModel("test_transformer_varied", self.sources, self.target, hyperparameters=self.hyperparams)
        attention_layers = [layer for layer in model.model.layers if isinstance(layer, keras.layers.MultiHeadAttention)]
        self.assertEqual(attention_layers[0].num_heads, 4)

if __name__ == '__main__':
    unittest.main()