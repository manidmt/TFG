'''
Testing Factorys/Models 

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import financial.momentum.storeLocalModel as  sLM
import financial.lab.evaluation as labevaluation


class ModelExperiment:

    ticker = None
    
    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead, **kwargs):
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

        sLM.storeLocal_data(ticker, self.model_factory, hyperparameters, self.name, self.datastore, self.lookahead, self.horizon)
        


class GlobalModelExperiment(ModelExperiment):

    def __init__(self, datastore, model_factory, name, start_year, end_year, lookahead=20):

        super().__init__(datastore, model_factory, name, start_year, end_year, lookahead)
        
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

        cross_validation = labevaluation.WalkForwardCrossValidation ( self.name, 
                                                                    hyperparameters, 
                                                                    features, 
                                                                    target, 
                                                                    df, 
                                                                    splits, 
                                                                    self.model_factory,
                                                                    save_path=os.environ["CACHE"],
                                                                    save_intermediate_results=False)
        cross_validation.run()
            
        # Final model

        print('Final model...')
            
        final_model = labevaluation.ModelTraining(self.model, hyperparameters, features, target, df, self.model_factory)
        final_model.run()









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
