'''
Testing Factorys/Models 

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import financial.data as fd
import financial.momentum.storeLocalModel as  sLM
import financial.lab.evaluation as labevaluation
import financial.lab.data as labdata
from sklearn.metrics import r2_score


class ModelExperiment:
    
    def __init__(self, datastore, ticker, model_factory, name:str, start_year:str, end_year:str, hyperparameters:dict, lookahead:int=None, horizon:int=None, **kwargs):
        self.datastore = datastore
        self.ticker = ticker
        self.model_factory = model_factory
        self.name = name
        self.start_year = start_year
        self.end_year = end_year
        self.hyperparameters = hyperparameters
        self.lookahead = lookahead
        self.horizon = horizon
        self.predictions = None


    def run(self):
        raise NotImplementedError("Subclasses must implement 'run'")







class LocalModelExperiment(ModelExperiment):

    def __init__(self, datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead=20, horizon=90):

        super().__init__(datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead, horizon)


    def run(self):
        self.predictions = sLM.storeLocal_data(self.ticker, self.model_factory, self.hyperparameters, self.name, self.datastore, None, self.start_year, self.end_year,  self.lookahead, self.horizon)
        # SE ESTÁ HACIENDO EL OUTPUT MANUALMENTE RELATIVE, COMPROBAR
        # MÉTRICAS CALCULAR CON EXPERIMENT --> SE NECESITA TARGET Y PREDICTION (TARGET = DATOS VERDADEROS)
class GlobalModelExperiment(ModelExperiment):

    def __init__(self, datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead=20, horizon=90):

        super().__init__(datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead, horizon)
        # Example: start_year = 1990, end_year = 2024, start_year_split = 1990 + (2024-1990)*5/7 = 2014
        start_year_split = round((int(end_year[:4]) - int(start_year[:4])) * 5/7) + int(start_year[:4])
        # Splits: 2014, 2015, 2016... 2023
        self.splits = [f"{year}-01-01" for year in range(start_year_split, int(end_year[:4]))]
        self.features = None
        self.target = None
        # Hiperparametros como parametro del constructor

        print(self.hyperparameters)

        
    def run(self): # Run sin paramtros, todo al constructor
        
        self.features = self.model_factory.input_descriptor(self.hyperparameters, self.datastore)
        self.target = self.model_factory.output_descriptor(self.hyperparameters, self.datastore)
        
        data_builder = labdata.DataStoreDataPreprocessing(self.name, self.ticker, self.datastore, self.features, self.target, self.start_year, self.end_year)
        data_builder.run()
        df = data_builder.dataset

        model = self.model_factory.create_model(self.name, self.hyperparameters, self.datastore)
        #print(model.to_xml_string())

        # print(df.columns)

        cross_validation = labevaluation.WalkForwardCrossValidation ( self.name, 
                                                                    self.hyperparameters, 
                                                                    self.features, 
                                                                    self.target, 
                                                                    df, 
                                                                    self.splits, 
                                                                    self.model_factory,
                                                                    save_path=os.environ["CACHE"],
                                                                    save_intermediate_results=False)
        cross_validation.run()
            
        # Final model
            
        final_model = labevaluation.ModelTraining(self.name, self.hyperparameters, self.features, self.target, df, self.model_factory)
        final_model.run()

        self.predictions =  reconstruct_relative_predictions_from_zscore(self.target, final_model.model.get_data(self.datastore, self.start_year, self.end_year))


def baseline_features(ds: fd.DataStore, hyperparameters: dict) -> fd.Set:
        features = fd.Set('Baseline features')
        
        ticker = hyperparameters["input"]["ticker"]
        horizon = hyperparameters["input"]["horizon"]

        variable = fd.Variable(ticker)
        #features.append(variable)
        for i in range(1, horizon+1): 
            features.append( fd.Change(variable, i) )

        return features

def reconstruct_relative_predictions_from_zscore(target, predictions):

    mean = target[0].mean
    stdev = target[0].stdev

    reconstructed_series = predictions * stdev + mean
    return reconstructed_series
class ModelExperimentFactory:
    
    @staticmethod
    def create_experiment(config):
        """
        Recibe un diccionario de configuración y devuelve una instancia del modelo correspondiente.
        """
        mode = config.get("mode")
        datastore = config.get("datastore")
        ticker = config.get("ticker")
        model_factory = config.get("model_factory")
        name = config.get("name")
        start_year = config.get("start_year")
        end_year = config.get("end_year")
        lookahead = config.get("lookahead", 20)
        horizon = config.get("horizon", 90)

        model_params = config.get("model_params", {})

        if mode == "local":
            hyperparameters = {
                "input": {
                    "features": "financial.momentum.storeLocalModel.local_features",
                    "horizon": horizon,
                    "ticker": ticker
                },
                "output": {
                    "target": [ticker],
                    "lookahead": lookahead,
                    "prediction": "relative"
                },
                    "model": model_params,
            }
            return LocalModelExperiment(datastore, ticker, model_factory, name, start_year, end_year, lookahead, horizon)

        elif mode == "global":
            hyperparameters = {
                "input": {
                    "features": "financial.momentum.experiment.modelExperiment.baseline_features",
                    "horizon": horizon,
                    "ticker": ticker,
                    "normalization": { "method": "z-score", "start_index": start_year, "end_index": end_year }
                    },
                "output": {
                    "target": [ticker],
                    "lookahead": lookahead,
                    "prediction": "relative",
                    "normalization": { "method": "z-score", "start_index": start_year, "end_index": end_year }
                    },
                "model": model_params,
            }
            return GlobalModelExperiment(datastore, ticker, model_factory, name, start_year, end_year, hyperparameters, lookahead, horizon)

        else:
            raise ValueError(f"Unknown mode: {mode}")
