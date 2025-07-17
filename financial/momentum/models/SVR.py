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
    """
    This class wraps a Scikit-Learn SVR model to provide a consistent interface for training and prediction.
    
    Methods:
        fit(X_train, y_train): Fit the model to the training data.
        predict(X): Make predictions using the trained model.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the SVR model to the training data.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        self.model.fit(X_train, y_train.values.ravel())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained SVR model.
        Args:
            X (pd.DataFrame): Input features for prediction.
        Returns:
            np.ndarray: Predicted values.
        """
        prediction = self.model.predict(X)

        # Ensure the prediction is 2D (n_samples, 1)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)

        return prediction

class SVRModelFactory(ModelFactory):
    """
    Factory class to create SVRModel instances.
    It follows the same structure as other model factories of the project.

    Methods:
        create_model_from_descriptors(model_id, hyperparameters, input_descriptor, output_descriptor):
            Creates a SVRModel instance from the provided descriptors and hyperparameters.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        """
        Creates a Scikit-Learn wrapped SVR model.
        Args:
            model_id (str): Unique identifier for the model.
            hyperparameters (dict): Hyperparameters for the model.
            input_descriptor (fd.DataDescriptor): Descriptor for input data.
            output_descriptor (fd.DataDescriptor): Descriptor for output data.
        Returns:
            SVRModel: An instance of SVRModel.
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