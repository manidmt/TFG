'''
Auxiliar Functions needed for the main code

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import pickle
import pandas as pd
import numpy as np

import financial.data as fd
import financial.model as fm
import financial.momentum.exponentialRegression as expReg
from financial.lab.models import ModelFactory
from sklearn.metrics import r2_score


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




def store_momentum_data(ticker, 
                        factory: ModelFactory,
                        hyperparameters: dict, 
                        model_name: str ="exponential",
                        ds: fd.DataStore = None, 
                        cache_path: str = None, 
                        lookahead: int = 20, 
                        horizon: int = 90):
    '''
    Computes and stores the momentum (Beta) and R² for a given ticker using FileCache.
    '''

    cache_path = os.environ.get("CACHE", "./cache") if cache_path is None else cache_path
    # Si no se pasa un DataStore, usar el definido por defecto
    if ds is None:
        from financial.io.cache import NoUpdateStrategy
        from financial.io.file.cache import FileCache

        
        ds = fd.CachedDataStore(path=os.environ["DATA"], 
                                cache=FileCache(cache_path=cache_path+"/", update_strategy=NoUpdateStrategy()))

    start_date = '1990-01-01'
    end_date = '2024-12-31'
    data = ds.get_data(ticker, start_date, end_date)
    #data = ds.get_data(ticker)  

    # Initialize series
    momentum_series = pd.Series(data=np.nan, index=data.index, dtype=float)
    r2_series = pd.Series(index=data.index)

    true_values = []
    predicted_values = []


    '''
    DEPURANDO CON PRINTS:
    '''

    forecast = pd.Series(index=data.index)

    for index in range(len(data) - horizon):
        model = create_local_model(factory, model_name, hyperparameters, ds, data, ticker, index, horizon)

        beta = model.model.coef_[0]  # Pendiente de la regresión exponencial
        # y_true = pd.Series([data.iloc[index + lookahead]])  # Valor real futuro
        # y_pred = pd.Series([model.predict([[lookahead]])])[0]  # Predicción del modelo
        forecast.iloc[index + horizon] = model.predict([[lookahead]])
        #print("Beta", beta)
        # true_values.append(y_true)
        # predicted_values.append(y_pred)
        # momentum_series.iloc[index + lookahead] = beta
        momentum_series.at[data.index[index + lookahead]] = beta
    '''    
        # print(f"Índice actual: {index}, Índice destino: {data.index[index + lookahead]}, Beta: {beta}")
        print(momentum_series.loc[data.index[index + lookahead]])
    

    print("Índice de momentum_series:", momentum_series.index[:10])  
    print("Índice de data:", data.index[:10])

    print("Tail 1 of momentum series before saving:\n", momentum_series.tail())

    print("True values: ", len(true_values))
    print("Predicted values: ", len(predicted_values))
    true_values = np.array(true_values).reshape(-1, 1)
    predicted_values = np.array(predicted_values).reshape(-1, 1)

    print("Estructura de true_values:", np.array(true_values).shape)
    print("Estructura de predicted_values:", np.array(predicted_values).shape)
    
    true_values = np.array(true_values).reshape(-1, 1)
    predicted_values = np.array(predicted_values).reshape(-1, 1)

    print("?? Primeros 10 valores de true_values:", true_values[:10].flatten())
    print("?? Primeros 10 valores de predicted_values:", predicted_values[:10].flatten())

    print("?? Ultimos 10 valores de true_values:", true_values[-10:].flatten())
    print("?? Ultimos 10 valores de predicted_values:", predicted_values[-10:].flatten())

    forecast = forecast.shift(lookahead)
    forecast = forecast.dropna() 
    forecast = forecast.copy()
    target = data[horizon+lookahead:]
    r2 = r2_score(target, forecast)
    true_values = np.array(true_values).reshape(-1)  # Convertir a numpy array y a 1D
    predicted_values = np.array(predicted_values).reshape(-1)  # Convertir también

    true_values = pd.Series(true_values, index=data.index[lookahead:])  # Asignar índice datetime
    predicted_values = pd.Series(predicted_values, index=data.index[:-lookahead])  # Asignar índice datetime

    predicted_series = predicted_values.shift(lookahead)  # Aplicamos el desplazamiento
    predicted_series = predicted_series.dropna()  # Eliminamos los valores NaN
    true_values = true_values[lookahead:]  # Ajustamos true_values para que coincida con predicted_series

    # ✅ Asegurar que los índices son compatibles
    print("🔍 Índices de true_values después del ajuste:", true_values.index[:10])
    print("🔍 Índices de predicted_series después del shift:", predicted_series.index[:10])

    # ✅ Ahora los índices deben ser iguales
    r2 = r2_score(true_values, predicted_series)
    '''

    forecast = forecast.shift(lookahead).dropna()
    target = data[horizon+lookahead:]
    r2 = r2_score(target, forecast)
    

    r2_series[:] = r2

    '''
    # Alternativa: convertir a diccionario y regenerar la serie
    momentum_dict = dict(momentum_series)  # Convertir a diccionario
    momentum_series = pd.Series(momentum_dict)  # Reconstruir la serie
    '''

    # Forzar actualizacion de la serie antes de guardarla
    momentum_series = momentum_series.dropna()  # Eliminar valores NaN
    momentum_series = momentum_series.copy()  # Asegurar que no es una vista de otra
    # Equiv =? momentum_series = momentum_series.dropna().copy() o momentum_series = pd.Series(momentum_series.dropna().to_dict())
   
    # Guardar las series en FileCache
   
    momentum_path = os.path.join(cache_path, f"model/momentum/{model_name}/{ticker}.pkl")
    r2_path = os.path.join(cache_path, f"model/momentum/{model_name}/{ticker}@r2.pkl")

    print("Tail 2 of momentum series before saving:\n", momentum_series.tail())

    momentum_dir = os.path.dirname(momentum_path)
    if not os.path.exists(momentum_dir):
        os.makedirs(momentum_dir)
    with open(momentum_path, 'wb') as file:
        pickle.dump(momentum_series, file)

    r2_dir = os.path.dirname(r2_path)
    if not os.path.exists(r2_dir):
        os.makedirs(r2_dir)
    with open(r2_path, 'wb') as file:
        pickle.dump(r2_series, file)




def local_regression_features(ds: fd.DataStore, ticker: str) -> fd.Set:
    '''
    Defines the input features for local regression models.
    '''
    features = fd.Set('Local autoregressive model features')    
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
            "features": "local_regression_features_wrapper"
            # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
            },
        "output": {
            "target": [ticker],
            "lookahead": lookahead,
            "prediction": "absolute" # "absolute"|"relative"
            # "normalization": { "method": "z-score", "start_index": start_date, "end_index": end_date }
            },    
    }

    store_momentum_data(ticker, expReg.ExponentialRegressionModelFactory(), hyperparameters, "exponential", lookahead=lookahead, ds=ds, cache_path=cache_path)



    