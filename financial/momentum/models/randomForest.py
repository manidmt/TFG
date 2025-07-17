'''
This module contains the RandomForestModelFactory class, 
which is used to create and train a Random Forest model for stock price prediction.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from sklearn.ensemble import RandomForestRegressor
import financial.model as fm
import pandas as pd
import numpy as np
import financial.data as fd
from financial.lab.models import ModelFactory



class RandomForestModel(fm.ScikitLearnModel):
    """
    This RandomForestModel class wraps a Scikit-Learn RandomForestRegressor model
    to provide a consistent interface for training and prediction.

    Methods:
        fit(X_train, y_train): Fit the model to the training data.
        predict(X): Make predictions using the trained model.
    """
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the Random Forest model to the training data.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        self.model.fit(X_train, y_train.values.ravel())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        Args:
            X (pd.DataFrame): Input features for prediction.
        Returns:
            np.ndarray: Predicted values.
        """
        prediction = self.model.predict(X)

        # Ensure the prediction is 2D (n_samples, 1), really it's extra having ravel()
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)

        return prediction

class RandomForestModelFactory(ModelFactory):
    """
    Factory class to create RandomForestModel instances.
    It follows the same structure as other model factories of the project.

    Methods:
        create_model_from_descriptors(model_id, hyperparameters, input_descriptor, output_descriptor):
            Creates a RandomForestModel instance from the provided descriptors and hyperparameters.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        """
        Creates a Scikit-Learn wrapped RandomForestRegressor model.
        Args:
            model_id (str): Unique identifier for the model.
            hyperparameters (dict): Hyperparameters for the model.
            input_descriptor (fd.DataDescriptor): Descriptor for input data.
            output_descriptor (fd.DataDescriptor): Descriptor for output data.
        Returns:
            RandomForestModel: An instance of RandomForestModel.
        """

        #  Default parameters for RandomForestModel
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Mix with the ones passed by hyperparameters
        params = default_params.copy()
        params.update(hyperparameters.get("model", {}))

        # Create scikit-learn model
        model = RandomForestRegressor(**params)

        # Return wrapped model
        return RandomForestModel(model_id, input_descriptor, output_descriptor, model, hyperparameters)
