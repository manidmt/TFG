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
    '''
    Exponential Regression Model based on Scikit-Learn
    
        Exponential Regression : Y = A*exp(Bx)
        Lienar Regression:       Y = A + Bx
        Solution:                ln(Y) = ln(A) + Bx
    '''

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        '''
        Ajust a linear regression model to the log of the target variable
        '''

        # Error al tratar con valores negativos e iguales a 0
        y_train_log = np.log(y_train)
        self.model.fit(X_train.values, y_train_log.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict the target variable using the exponential regression model
        '''
        y_pred_log = self.model.predict(X.values)  
        return np.exp(y_pred_log)


class ExponentialRegressionModelFactory(ModelFactory):
    '''
    Exponential regression model factory
    '''

    def create_model_from_descriptors(self, 
                                      model_id: str, 
                                      hyperparameters: dict, 
                                      input_descriptor: fd.DataDescriptor, 
                                      output_descriptor: fd.DataDescriptor) -> fm.Model:
        # Usamos regresión lineal sobre los logaritmos de Y
        model = sklearn.linear_model.LinearRegression()
        
        return ExponentialScikitLearnModel(model_id, input_descriptor, output_descriptor, model, hyperparameters)




'''
Test
'''

if __name__ == "__main__":
    print("Ejecutando prueba de ExponentialRegressionModelFactory...")

    import pandas as pd

    # Datos de prueba (X = días, Y = precios)
    X_train = pd.DataFrame({'dias': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y_train = pd.Series([100, 105, 110, 120, 130, 150, 170, 200, 240, 290])

    # Crear modelo
    factory = ExponentialRegressionModelFactory()
    model = factory.create_model_from_descriptors("exp_model", {}, None, None)

    # Entrenar modelo
    model.fit(X_train, y_train)

    # Hacer una predicción
    X_test = pd.DataFrame({'dias': [11, 12, 13, 14, 15]})
    y_pred = model.predict(X_test)

    print("Predicciones:", y_pred)
