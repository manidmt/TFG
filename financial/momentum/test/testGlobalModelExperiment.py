'''
Testing GlobalModelExperiment class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed (TensorFlow)

import pandas as pd
import financial.data as fd

from financial.io.file.cache import FileCache
from financial.momentum.experiment.modelExperiment import *
from financial.momentum.exponentialRegression import ExponentialRegressionModelFactory

class TestGlobalModelExperiment(unittest.TestCase):

    def setUp(self):
        self.datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/"))
        self.factory = ExponentialRegressionModelFactory()
        self.name = "global-test"
        self.start_year = "2010-01-01"
        self.end_year = "2024-01-01"
        self.lookahead = 20
        self.ticker = 'AAPL'

    def test_run(self):
        global_model_experiment = GlobalModelExperiment(self.datastore, self.factory, self.name, self.start_year, self.end_year, self.lookahead)
        global_model_experiment.run(self.ticker)
        prediction_experiment = global_model_experiment.predictions * 100
        # Verificar que la salida no es NaN y tiene datos
        self.assertIsInstance(prediction_experiment, pd.Series)
        self.assertFalse(prediction_experiment.isna().all())

if __name__ == '__main__':
    unittest.main()