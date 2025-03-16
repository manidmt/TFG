'''
Testing LocalModelExperiment class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''


import unittest
import os

import pandas as pd
import numpy as np
import financial.model as fm
import financial.data as fd

from financial.io.file.cache import FileCache
from financial.momentum.exponentialRegression import ExponentialRegressionModelFactory
from financial.momentum.experiment.modelExperiment import LocalModelExperiment
from financial.momentum.storeLocalModel import storeLocal_data

class TestLocalModelExperiment(unittest.TestCase):

    def setUp(self):
        self.datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/"))
        self.factory = ExponentialRegressionModelFactory()
        self.name = 'Test'
        self.start_year = '1990-01-01'
        self.end_year = '2023-12-31'
        self.lookahead = 20
        self.horizon = 90
        self.ticker = 'AAPL'

    def test_run(self):

        local_model_experiment = LocalModelExperiment(self.datastore, self.factory, self.name, self.start_year, self.end_year, self.lookahead, self.horizon)
        
        print(f"🔍 start_year: {self.start_year}, type: {type(self.start_year)}")
        print(f"🔍 end_year: {self.end_year}, type: {type(self.end_year)}")

        
        local_model_experiment.run(self.ticker)
        prediction_experiment = local_model_experiment.predictions * 100

        prediction_manual = self.manual_model_creation(self.ticker, self.datastore, self.factory, self.start_year, self.end_year, self.lookahead, self.horizon) * 100

        np.testing.assert_almost_equal(prediction_experiment, prediction_manual, decimal=2) # Está bien el manual?
    
    def manual_model_creation(self, ticker, datastore, factory, start_date, end_date, lookahead, horizon):

        data = self.datastore.get_data(ticker, start_date, end_date)
        experiment_id = 'exponential-regression-model'

        hyperparameters = {
                "input": {
                    "features": "local_regression_features_wrapper"
                    # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
                    },
                "output": {
                    "target": [ticker],
                    "lookahead": lookahead,
                    "prediction": "relative" # "absolute"|"relative"
                    # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
                    },    
        }

        
        
        
        def create_local_model(start_index: int=0, samples: int=horizon) -> fm.Model:
            model = factory.create_model(experiment_id, hyperparameters, datastore)
            series = data[start_index:start_index+samples]
            subset = pd.DataFrame(series, index = data.index[start_index:start_index+samples])
            subset['day'] = range(-len(subset)+1,1)
            model.fit(subset[['day']],subset[self.ticker])
    

            return model

        def local_regression(data, samples: int=self.horizon):
            forecast = pd.Series(index=data.index)
            for index in range(len(data)-samples):
                model = create_local_model(index, samples)
                forecast.iloc[index+samples] = model.predict([[lookahead]])
                forecast.iloc[index+samples] = forecast.iloc[index+samples] / data.iloc[index+samples]
            return forecast

        forecast = local_regression(data, samples=self.horizon)
        return forecast


    def local_regression_features(ds: fd.DataStore, ticker: str) -> fd.Set:
                features = fd.Set('Local exponential regression model features')    
                variable = fd.Variable(ticker)
                features.append(variable)
                return features

    def local_regression_features_wrapper(self, ds: fd.DataStore) -> fd.Set:
                return self.local_regression_features(ds,self.ticker)  # ¿COMO FUNCIONAN LOS WRAPPERS?
     
    
if __name__ == '__main__':
    unittest.main()