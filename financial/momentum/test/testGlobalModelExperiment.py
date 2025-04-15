'''
Testing GlobalModelExperiment class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed (TensorFlow)

import pandas as pd
import financial.data as fd
import numpy as np

from financial.io.file.cache import FileCache
from financial.momentum.experiment.modelExperiment import *
from financial.momentum.models.exponentialRegression import ExponentialRegressionModelFactory

class TestGlobalModelExperiment(unittest.TestCase):

    def setUp(self):
        self.datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/"))
        self.factory = ExponentialRegressionModelFactory()
        self.name = "global-test"
        self.start_year = "1990-01-01"
        self.end_year = "2023-12-31"
        self.lookahead = 20
        self.horizon = 90
        self.ticker = '^GSPC'
        self.model_params = None

        self.hyperparameters = {
            "input": {
                "features": "financial.momentum.experiment.modelExperiment.baseline_features",
                "horizon": self.horizon,
                "ticker": self.ticker,
                "normalization": { "method": "z-score", "start_index": self.start_year, "end_index": self.end_year }
                },
            "output": {
                "target": [self.ticker],
                "lookahead": self.lookahead,
                "prediction": "relative",
                "normalization": { "method": "z-score", "start_index": self.start_year, "end_index": self.end_year }
                },
            "model": self.model_params,
        }

        global_model_experiment = GlobalModelExperiment(self.datastore, self.ticker, self.factory, self.name, self.start_year, self.end_year, self.hyperparameters, self.lookahead)
        global_model_experiment.run()
        self.experiment = global_model_experiment

    def test_predictions_are_series_not_na(self):
        prediction_experiment = self.experiment.predictions
        self.assertIsInstance(prediction_experiment, pd.Series)
        self.assertFalse(prediction_experiment.isna().all())

    def test_predictions_valid_date(self):
        index = self.experiment.predictions.index
        self.assertTrue(index[0] >= pd.to_datetime(self.start_year))
        self.assertTrue(index[-1] <= pd.to_datetime(self.end_year))

    def test_ticker_not_valid(self):
        with self.assertRaises(KeyError):
            self.hyperparameters["input"]["ticker"] = "INVALID_TICKER"
            self.hyperparameters["output"]["target"] = ["INVALID_TICKER"]
            invalid_model_experiment = GlobalModelExperiment(self.datastore, "INVALID_TICKER", self.factory, self.name, self.start_year, self.end_year, self.hyperparameters, self.lookahead)
            invalid_model_experiment.run()
    
    def test_date_not_valid(self):
        with self.assertRaises(ValueError):
            invalid_model_experiment = GlobalModelExperiment(self.datastore, self.ticker, self.factory, self.name, "INVALID_DATE", "INVALID_DATE", self.hyperparameters, self.lookahead)
            invalid_model_experiment.run()

    def test_horizon_not_valid(self):
        with self.assertRaises(ValueError):
            self.hyperparameters["input"]["horizon"] = -1
            invalid_model_experiment = GlobalModelExperiment(self.datastore, self.ticker, self.factory, self.name, self.start_year, self.end_year, self.hyperparameters, self.lookahead, horizon=-1)
            invalid_model_experiment.run()

    def test_complete(self):
        ds = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/"))
        start_date = '1990-01-01'
        end_date = '2023-12-31'
        ticker = '^GSPC'
        data = ds.get_data(ticker, start_date, end_date)

        experiment_id = 'test-model'
        lookahead = 20 # i.e. ~ 1 mes (4 semanas)
        horizon   = 90 # i.e. ~ 1 mes (4 semanas)

        hyperparameters = {
                "input": {
                    "features": "financial.momentum.experiment.modelExperiment.baseline_features",
                    "horizon": horizon,
                    "ticker": ticker,
                    "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
                    },
                "output": {
                    "target": [ticker],
                    "lookahead": lookahead,
                    "prediction": "relative", # "absolute"|"relative"
                    "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
                    },    
        }

        factory = ExponentialRegressionModelFactory()
        features = factory.input_descriptor(hyperparameters, ds) # inputs|standardized_inputs
        target = factory.output_descriptor(hyperparameters, ds) # outputs|change_outputs|standardized_outputs

        data_builder = labdata.DataStoreDataPreprocessing(experiment_id, ticker, ds, features, target, start_date, end_date)
        data_builder.run()
        df = data_builder.dataset

        splits = [ '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', 
           '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
        cross_validation = labevaluation.WalkForwardCrossValidation ( experiment_id, 
                                                                    hyperparameters, 
                                                                    features, 
                                                                    target, 
                                                                    df, 
                                                                    splits, 
                                                                    factory,
                                                                    save_path=os.environ["CACHE"],
                                                                    save_intermediate_results=False)
        cross_validation.run()

        final_model = labevaluation.ModelTraining(experiment_id, hyperparameters, features, target, df, factory)
        final_model.run()
        model_output = final_model.model.get_data(ds, start_date, end_date)

        mean = target[0].mean
        stdev = target[0].stdev

        reconstructed_series = model_output * stdev + mean

        np.testing.assert_almost_equal(reconstructed_series.to_numpy(), self.experiment.predictions.to_numpy(), decimal=2)

if __name__ == '__main__':
    unittest.main()