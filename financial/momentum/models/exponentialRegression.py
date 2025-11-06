'''
Exponential Regression -- Clenow

@author: Manuel Díaz-Meco (manidmt5@gmail.com) 
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed (TensorFlow)

# from financial.model import Model
from financial.lab.models import ModelFactory


import financial.data as fd
import financial.model as fm

import numpy as np
import pandas as pd
import sklearn.linear_model

class ExponentialScikitLearnModel(fm.ScikitLearnModel):
    """
    Exponential Regression Model based on Scikit-Learn
    
        Exponential Regression : Y = A*exp(Bx)
        Lienar Regression:       Y = A + Bx
        Solution:                ln(Y) = ln(A) + Bx

    Methods:
        fit(X_train, y_train): Fit the model to the training data
        predict(X): Predict the target variable using the fitted model
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit a linear regression model to the log of the target variable
        Args:
            X_train (pd.DataFrame): Training data features
            y_train (pd.Series): Training data target variable
        """
        if "normalization" in  self.hyperparameters["output"]:
            y_train = y_train + abs(y_train.min()) + 1  # Normalization to avoid 0 or negative values [We won't normalize using this model]

        # Convert y_train to log scale, if is a non-negative value it will raise an error as expected
        # This model is the first approach of the project, the values given to the model won't be negative
        # as we do not normalize the data and financial times series values are always positive
        y_train_log = np.log(y_train)
        self.model.fit(X_train.values, y_train_log.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target variable using the exponential regression model
        Args:
            X (pd.DataFrame): Input features for prediction
        Returns:
            np.ndarray: Predicted values in the original scale (exponential)
        """
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        y_pred_log = self.model.predict(X.values)  
        return np.exp(y_pred_log)

class ExponentialRegressionModelFactory(ModelFactory):
    """
    Factory class to create ExponentialScikitLearnModel instances.
    It follows the same structure as other model factories of the project.

    Methods:
        create_model_from_descriptors(model_id, hyperparameters, input_descriptor, output_descriptor):
            Creates an ExponentialScikitLearnModel instance from the provided descriptors and hyperparameters.
    """

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        
        model = sklearn.linear_model.LinearRegression()
        
        return ExponentialScikitLearnModel(model_id, input_descriptor, output_descriptor, model, hyperparameters)


"""
Test ExponentialRegressionModelFactory
"""

if __name__ == "__main__":
    print("Testing ExponentialRegressionModelFactory...")

    import pandas as pd

    # Test data (X = days, Y = prices)
    X_train = pd.DataFrame({'days': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y_train = pd.Series([100, 105, 110, 120, 130, 150, 170, 200, 240, 290])

    # Create model
    factory = ExponentialRegressionModelFactory()
    model = factory.create_model_from_descriptors("exp_model", {}, None, None)

    # Train model
    model.fit(X_train, y_train)

    # Make a prediction
    X_test = pd.DataFrame({'days': [11, 12, 13, 14, 15]})
    y_pred = model.predict(X_test)

    print("Predictions:", y_pred)
