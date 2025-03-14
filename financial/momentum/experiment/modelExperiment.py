'''
Testing Factorys/Models 

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import financial.momentum.storeLocalModel as  sLM
import financial.lab.evaluation as labevaluation
import financial.data as labdata


class ModelExperiment:

    ticker = None
    
    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=None, **kwargs):
        self.datastore = datastore
        self.model_factory = model_factory
        self.name = name
        self.start_year = start_year
        self.end_year = end_year
        self.lookahead = lookahead

    
    def run(self, ticker):
        raise NotImplementedError("Subclasses must implement 'run'")







class LocalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20, horizon=90):

        super().__init__(datastore, model_factory, name, start_year, end_year, lookahead)
        self.horizon = horizon


    def run(self, ticker):
        
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

        forecast = sLM.storeLocal_data(ticker, self.model_factory, hyperparameters, self.name, self.datastore, self.lookahead, self.horizon)
        return forecast


class GlobalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20):

        super().__init__(datastore, model_factory, name, start_year, end_year, lookahead)
        # Example: start_year = 1990, end_year = 2024, start_year_split = 1990 + (2024-1990)*5/7 = 2014
        start_year_split = round((int(end_year[:4]) - int(start_year[:4])) * 5/7) + int(start_year[:4])
        # Splits: 2014, 2015, 2016... 2023
        self.splits = [f"{year}-01-01" for year in range(start_year_split, int(end_year[:4]))]
        
    def run(self, ticker):
        
        hyperparameters = {
        "input": {
            "features": "baseline_features_wrapper",
            },
        "output": {
            "target": [ticker],
            "lookahead": self.lookahead,
            "prediction": "relative", # "absolute"|"relative"
            },    
        }
        features = self.model_factory.input_descriptor(hyperparameters, self.datastore)
        target = self.model_factory.output_descriptor(hyperparameters, self.datastore)
        
        data_builder = labdata.DataStoreDataPreprocessing(self.name, ticker, self.datastore, features, target, self.start_year, self.end_year)
        data_builder.run()
        df = data_builder.dataset


        cross_validation = labevaluation.WalkForwardCrossValidation ( self.name, 
                                                                    hyperparameters, 
                                                                    features, 
                                                                    target, 
                                                                    df, 
                                                                    self.splits, 
                                                                    self.model_factory,
                                                                    save_path=os.environ["CACHE"],
                                                                    save_intermediate_results=False)
        cross_validation.run()
            
        # Final model

        print('Final model...')
            
        final_model = labevaluation.ModelTraining(self.name, hyperparameters, features, target, df, self.model_factory)
        final_model.run()

        return final_model.model.get_data(self.datastore, self.start_year, self.end_year)









class ModelExperimentFactory:
    
    @staticmethod
    def create_experiment(config):
        """
        Recibe un diccionario de configuración y devuelve una instancia del modelo correspondiente.
        """
        mode = config.get("mode")
        datastore = config.get("datastore")
        model_factory = config.get("model_factory")
        name = config.get("name")
        start_year = config.get("start_year")
        end_year = config.get("end_year")

        if mode == "local":
            lookahead = config.get("lookahead", 20)
            horizon = config.get("horizon", 90)
            return LocalModelExperiment(datastore, model_factory, name, start_year, end_year, lookahead, horizon)

        elif mode == "global":
            return GlobalModelExperiment(datastore, model_factory, name, start_year, end_year)

        else:
            raise ValueError(f"Unknown mode: {mode}")
