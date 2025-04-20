'''
This module implements a Support Vector Regression (SVR) model using the scikit-learn library.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from sklearn.svm import SVR
import financial.model as fm
import financial.data as fd
import pandas as pd
import numpy as np
from financial.lab.models import ModelFactory

class SVRModel(fm.ScikitLearnModel):
    '''
    Support Vector Regression Model adapted to the financial framework.
    '''

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train.values.ravel())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(X)

        # Ensure the prediction is 2D (n_samples, 1)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)

        return prediction

class SVRModelFactory(ModelFactory):
    """
    Support Vector Regression model factory.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        """
        Creates a Scikit-Learn wrapped SVR model.
        """

        # Default hyperparameters
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }

        # Merge with the ones passed in hyperparameters
        params = default_params.copy()
        params.update(hyperparameters.get("model", {}))

        # Create scikit-learn model
        model = SVR(**params)

        return SVRModel(model_id, input_descriptor, output_descriptor, model, hyperparameters)