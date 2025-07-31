'''
Testing RecurrentModel class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import financial.data as fd
import numpy as np

from financial.momentum.models.kerasAdvanced import RecurrentModel

class TestRecurrentModel(unittest.TestCase):
    
    def setUp(self):
        self.horizon = 10
        self.variables = ["X1", "X2"]
        self.sources = fd.Set("input")
        for var in self.variables:
            self.sources.append(fd.Variable(var))
        self.target = fd.Variable("Y")
        self.hyperparameters = {
            "model": {"architecture": "rnn"},
            "topology": {
                "layers": [32, 16],
                "activation": {"output": "linear"}
            },
            "input": {"horizon": self.horizon},
            "optimization": {
                "optimizer": "adam",
                "loss": "mean_squared_error",
                "metrics": ["mae"],
                "epochs": 1,
                "batch_size": 8,
                "validation_split": 0.1
            }
        }
        self.model = RecurrentModel("testRNN", self.sources, self.target, hyperparameters=self.hyperparameters)

    def test_reshape_input_shape(self):
        X = np.random.rand(100, len(self.variables) * self.horizon)
        X_reshaped = self.model.reshape_input(X)
        self.assertEqual(X_reshaped.shape, (100, self.horizon, len(self.variables)))

    def test_reshape_input_invalid_shape(self):
        X = np.random.rand(100, 23)
        with self.assertRaises(AssertionError):
            self.model.reshape_input(X)

    def test_initialize_model_shape(self):
        model = self.model.initialize_model()
        input_shape = model.input_shape
        self.assertEqual(input_shape, (None, self.horizon, len(self.variables)))

    def test_fit_and_predict(self):
        X = np.random.rand(100, len(self.variables) * self.horizon)
        y = np.random.rand(100, 1)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.assertEqual(preds.shape, (100, 1))
    
    def test_reshape_input_with_dataframe(self):
        df = pd.DataFrame(np.random.rand(50, len(self.variables) * self.horizon))
        reshaped = self.model.reshape_input(df)
        self.assertEqual(reshaped.shape, (50, self.horizon, len(self.variables)))

    def test_number_of_trainable_parameters(self):
        model = self.model.model
        total_params = model.count_params()
        expected_input_units = self.horizon * len(self.variables)
        self.assertGreater(total_params, expected_input_units)

    def test_mismatched_horizon_and_input_shape(self):
        bad_hyper = self.hyperparameters.copy()
        bad_hyper["input"] = {"horizon": 7} 
        bad_model = RecurrentModel("badModel", self.sources, self.target, hyperparameters=bad_hyper)
        with self.assertRaises(AssertionError):
            X_bad = np.random.rand(10, 20)
            bad_model.reshape_input(X_bad)

    def test_output_layer_shape(self):
        model = self.model.model
        output_shape = model.output_shape
        self.assertEqual(output_shape[-1], 1)

    def test_fit_with_extreme_values(self):
        X = np.random.rand(50, len(self.variables) * self.horizon)
        X[0][0] = np.nan
        y = np.random.rand(50, 1)
        with self.assertRaises(ValueError):
            self.model.fit(X, y)


if __name__ == '__main__':
    unittest.main()