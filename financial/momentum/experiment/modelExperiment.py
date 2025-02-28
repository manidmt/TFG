'''
Testing Factorys/Models 

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import financial.momentum.storeLocalModel as  sLM


class ModelExperiment:
    
    def __new__(cls, mode, datastore, model_factory, name, start_year, end_year,  **kwargs):

        if mode == 'local':
            return LocalModelExperiment() 
        elif mode == 'global':
            return GlobalModelExperiment()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def __init__(self, datastore, model_factory, name, start_year, end_year, **kwargs):
        self.datastore = datastore
        self.model_factory = model_factory
        self.name = name
        self.start_year = start_year
        self.end_year = end_year   






class LocalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20, horizon=90):

        super().__init__(datastore, model_factory, name, start_year, end_year)
        self.lookahead = lookahead
        self.horizon = horizon


    def run(self, ticker):

        def local_feature_wrapper(data):

            return sLM.local_features(data, ticker)
        
        hyperparameters = {
            "input":{
                "features": "local_features_wrapper"
            },
            "output":{
                "target": [ticker],
                "lookahead": self.lookahead,
                "prediction": "relative"
            }
        }

        sLM.storeLocal_data(ticker, self.model_factory, hyperparameters, self.name, self.datastore, self.lookahead, self.horizon)
        


class GlobalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory):

        super().__init__(datastore, model_factory)
        
