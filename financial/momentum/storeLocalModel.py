'''
Storing and computing local model data for a given ticker.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import pickle
import pandas as pd
import numpy as np

import financial.data as fd
import financial.model as fm
import financial.momentum.models.exponentialRegression as expReg
from financial.lab.models import ModelFactory
from sklearn.metrics import r2_score
from financial.io.cache import NoUpdateStrategy
from financial.io.file.cache import FileCache


def create_local_model(factory: ModelFactory,
                       experiment_id: str,
                       hyperparameters: dict,
                       ds: fd.DataStore,
                       data: pd.Series,
                       ticker: str,
                       start_index: int = 0,
                       samples: int = 90) -> fm.Model:
    '''
    Creates a local regression model based on the last `samples` days.
    '''
    model = factory.create_model(experiment_id, hyperparameters, ds)

    # Select the data window
    series = data[start_index:start_index+samples]
    subset = pd.DataFrame(series, index=data.index[start_index:start_index+samples])
    subset['day'] = range(-len(subset)+1, 1)  # Create a time index

    # Fit the model
    model.fit(subset[['day']], subset[ticker])

    return model




def storeLocal_data(ticker, 
                        factory: ModelFactory,
                        hyperparameters: dict, 
                        model_name: str ="exponential",
                        ds: fd.DataStore = None, 
                        cache_path: str = None, 
                        start_date: str = '1990-01-01',
                        end_date: str = '2024-12-31',
                        lookahead: int = 20, 
                        horizon: int = 90,
                        store_slope: bool = False,
                        store_r2: bool = False):
    '''
    Computes and stores the momentum (Beta) and R² for a given ticker using FileCache.
    '''

    cache_path = os.environ.get("CACHE", "./cache") if cache_path is None else cache_path
    # Si no se pasa un DataStore, usar el definido por defecto
    if ds is None:
        
        ds = fd.CachedDataStore(path=os.environ["DATA"], 
                                cache=FileCache(cache_path=cache_path+"/", update_strategy=NoUpdateStrategy()))


    start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
    end_date = pd.Timestamp(end_date) if isinstance(end_date, str) else end_date

    data = ds.get_data(ticker, start_date, end_date)


    # Initialize series
    slope_series = pd.Series(data=np.nan, index=data.index, dtype=float) if store_slope else None
    r2_series = pd.Series(index=data.index)                                 if store_r2 else None       
    forecast = pd.Series(index=data.index)
    relative_predicted_values = pd.Series(index=data.index)

    for index in range(len(data) - horizon):

        model = create_local_model(factory, model_name, hyperparameters, ds, data, ticker, index, horizon)
                                        
        forecast.iloc[index + horizon] = model.predict([[lookahead]])
        # If we want a more clearly view, we should multiply by 100:
        relative_predicted_values[data.index[index+horizon]] = ((forecast.iloc[index + horizon] / data.iloc[index + horizon])) - 1 if data.iloc[index + horizon] != 0 else 0

        if store_slope:
            beta = model.model.coef_[0]                             # Slope of the regression line 
            slope_series.at[data.index[index + horizon]] = beta

        if store_r2:
            if index >= horizon:
                X_train = pd.DataFrame(np.arange(-horizon + 1, 1).reshape(-1, 1))  # Generamos la matriz de días usados en la regresión
                y_true = data.iloc[index: index + horizon]  # Valores reales usados en la regresión
                y_pred = model.predict(X_train)  # Predicciones del modelo en los datos de entrenamiento

                
                y_true_clean = y_true.dropna()
                y_pred_clean = pd.Series(y_pred, index=y_true.index).dropna() 

                if len(y_true_clean) > 1 and len(y_pred_clean) > 1:
                    r2_series.at[data.index[index + horizon]] = r2_score(y_true_clean, y_pred_clean)

    
    relative_predicted_values = relative_predicted_values.dropna()
    # Save the series
    prediction_path = os.path.join(cache_path, f"model_momentum-{model_name}-{ticker}")

    store_results(prediction_path, ticker, model_name, relative_predicted_values, slope_series, r2_series)
    '''
    with open(prediction_path, 'wb') as file:
        pickle.dump(relative_predicted_values, file)
    
    if store_slope:
        # Forzar actualizacion de la serie antes de guardarla
        slope_series = slope_series.dropna()  # Eliminar valores NaN
        slope_series = slope_series.copy()  # Asegurar que no es una vista de otra
        slope_path = os.path.join(cache_path, f"model-momentum-{model_name}-{ticker}@slope")
        with open(slope_path, 'wb') as file:
            pickle.dump(slope_series, file)
    
    if store_r2:
        r2_series = r2_series.dropna()
        r2_series = r2_series.copy()
        r2_path = os.path.join(cache_path, f"model-momentum-{model_name}-{ticker}@r2")
        with open(r2_path, 'wb') as file:
            pickle.dump(r2_series, file)
    
    '''
    return relative_predicted_values


def store_results(prediction_path: str, ticker: str, model_name: str, predictions: pd.Series, slope_series: pd.Series = None, r2_series: pd.Series = None):
    with open(prediction_path, 'wb') as file:
        pickle.dump(predictions, file)
    
    if slope_series is not None:
        slope_series = slope_series.dropna()
        slope_series = slope_series.copy()
        slope_path = os.path.join(prediction_path, f"model-momentum-{model_name}-{ticker}@slope")
        with open(slope_path, 'wb') as file:
            pickle.dump(slope_series, file)

    if r2_series is not None:
        r2_series = r2_series.dropna()
        r2_series = r2_series.copy()
        r2_path = os.path.join(prediction_path, f"model-momentum-{model_name}-{ticker}@r2")
        with open(r2_path, 'wb') as file:
            pickle.dump(r2_series, file)

    # model/momentum/{model_name}/{ticker} no funcionaba --> cambio a -  ¿solución modificando el DataStore?


def local_features(ds: fd.DataStore, ticker: str) -> fd.Set:
    '''
    Defines the input features for local regression models.
    '''
    features = fd.Set('Local model features')    
    variable = fd.Variable(ticker)
    features.append(variable)
    return features



'''
def local_regression_features_wrapper(ds: fd.DataStore, ticker: str) -> fd.Set:
    ''
    Wrapper function to generate local regression features.
    This is required by ModelFactory when creating a model.
    ''
    return local_regression_features(ds, ticker)
'''


def store_exponentialModel_data(ticker,
                                ds: fd.DataStore = None, 
                                cache_path: str = None,
                                lookahead: int = 20):
    

    hyperparameters = {
        "input": {
            "features": "local_features_wrapper"
            # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
            },
        "output": {
            "target": [ticker],
            "lookahead": lookahead,
            "prediction": "absolute" # "absolute"|"relative"
            # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
            },    
    }

    storeLocal_data(ticker, expReg.ExponentialRegressionModelFactory(), hyperparameters, "exponential", ds, cache_path, lookahead=lookahead, store_slope=True, store_r2=True)



    