'''
This module contains the RandomForestModelFactory class, 
which is used to create and train a Random Forest model for stock price prediction.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from sklearn.ensemble import RandomForestRegressor
import financial.model as fm
import financial.data as fd
from financial.lab.models import ModelFactory

class RandomForestModelFactory(ModelFactory):
    """
    Random Forest regression model factory.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        """
        Creates a Scikit-Learn wrapped RandomForestRegressor model.
        """

        # Hiperparámetros por defecto
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Mezclar con los que se pasen por hyperparameters
        params = default_params.copy()
        params.update(hyperparameters.get("model", {}))

        # Crear modelo scikit-learn
        model = RandomForestRegressor(**params)

        # Devolver modelo envuelto
        return fm.ScikitLearnModel(model_id, input_descriptor, output_descriptor, model, hyperparameters)
