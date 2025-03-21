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
    
    def __init__(self, datastore, model_factory, name:str, start_year:str, end_year:str, lookahead:int=None, horizon:int=None, **kwargs):
        self.datastore = datastore
        self.model_factory = model_factory
        self.name = name
        self.start_year = start_year
        self.end_year = end_year
        self.lookahead = lookahead
        self.horizon = horizon

        self.metrics = {

            "MSE": None,
            "RMSE": None,
            "MAE": None,
            "MAPE": None,
            "R2": None
        }
        self.predictions = None
        self.ticker = None


    def run(self, ticker):
        raise NotImplementedError("Subclasses must implement 'run'")







class LocalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20, horizon=90):

        super().__init__(datastore, model_factory, name, start_year, end_year, lookahead, horizon)


    def run(self, ticker):

        self.ticker = ticker
        
        self.hyperparameters = {
            "input":{
                "features": "local_features_wrapper"
            },
            "output":{
                "target": [self.ticker],
                "lookahead": self.lookahead,
                "prediction": "relative"
            }
        }

        self.predictions = sLM.storeLocal_data(self.ticker, self.model_factory, self.hyperparameters, self.name, self.datastore, None, self.start_year, self.end_year,  self.lookahead, self.horizon)
        # SE ESTÁ HACIENDO EL OUTPUT MANUALMENTE RELATIVE, COMPROBAR
        # MÉTRICAS CALCULAR CON EXPERIMENT --> SE NECESITA TARGET Y PREDICTION (TARGET = DATOS VERDADEROS)

GLOBAL_HORIZON = 20
ticker_global = None
class GlobalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20, horizon=GLOBAL_HORIZON):

        super().__init__(datastore, model_factory, name, start_year, end_year, lookahead, horizon)
        # Example: start_year = 1990, end_year = 2024, start_year_split = 1990 + (2024-1990)*5/7 = 2014
        start_year_split = round((int(end_year[:4]) - int(start_year[:4])) * 5/7) + int(start_year[:4])
        # Splits: 2014, 2015, 2016... 2023
        self.splits = [f"{year}-01-01" for year in range(start_year_split, int(end_year[:4]))]

        self.hyperparameters = None
        self.features = None
        self.target = None
        # Hiperparametros como parametro del constructor

        
    def run(self, ticker): # Run sin paramtros, todo al constructor

        self.ticker = ticker
        ticker_global = ticker
        
        
        self.hyperparameters = {
            "input": {
                "features": "financial.momentum.experiment.modelExperiment.baseline_features",
                "horizon": self.horizon,
                "ticker": self.ticker
                },
            "output": {
                "target": [ticker],
                "lookahead": self.lookahead,
                "prediction": "relative", # "absolute"|"relative"
                },    
        }
        self.features = self.model_factory.input_descriptor(self.hyperparameters, self.datastore)
        self.target = self.model_factory.output_descriptor(self.hyperparameters, self.datastore)
        
        data_builder = labdata.DataStoreDataPreprocessing(self.name, self.ticker, self.datastore, self.features, self.target, self.start_year, self.end_year)
        data_builder.run()
        df = data_builder.dataset


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

        self.metrics["MSE"] = final_model.results[self.ticker].MSE()
        self.metrics["RMSE"] = final_model.results[self.ticker].RMSE()
        self.metrics["MAE"] = final_model.results[self.ticker].MAE()
        self.metrics["MAPE"] = final_model.results[self.ticker].MAPE() # NO HACE FALTA --> MÁS ADELANTE, DEVOLVER TARGET Y PREDICTION
        self.metrics["R2"] = self.R2()

        self.predictions =  final_model.model.get_data(self.datastore, self.start_year, self.end_year)


    def reconstruct_from_relative(self, predictions):

        mean = self.target[0].mean
        stdev = self.target[0].stdev 
        data = self.datastore.get_data(self.ticker, self.start_year, self.end_year)

        reconstructed_change = - (mean + stdev * predictions)
        reconstructed_final = data / (1 - reconstructed_change)

        return reconstructed_final.shift(self.lookahead).dropna()


    def R2(self):

        r2 = r2_score(self.predictions, self.reconstruct_from_relative(self.predictions))
        return r2

    def get_metrics(self):
        return self.metrics



def baseline_features(ds: fd.DataStore, hyperparameters: dict) -> fd.Set:
        features = fd.Set('Baseline features')
        
        ticker = hyperparameters["input"]["ticker"]
        horizon = hyperparameters["input"]["horizon"]

        variable = fd.Variable(ticker)
        #features.append(variable)
        for i in range(1, horizon+1): 
            features.append( fd.Change(variable, i) )

        return features

def baseline_features_wrapper(ds: fd.DataStore) -> fd.Set:
            return baseline_features(ds,ticker_global)


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
