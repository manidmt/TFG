'''
Auxiliar Functions needed for the main code

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os
import pickle
import pandas as pd

import financial.data as fd
import financial.model as fm
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
                        model_name="exponential",
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

    data = ds.get_data(ticker)  

    # Initialize series
    momentum_series = pd.Series(index=data.index)
    r2_series = pd.Series(index=data.index)

    for index in range(len(data) - horizon):
        model = create_local_model(factory, model_name, hyperparameters, ds, data, ticker, index, horizon)

        beta = model.model.coef_[0]  # Pendiente de la regresión exponencial
        y_true = pd.Series([data.iloc[index + lookahead]])  # Valor real futuro
        y_pred = pd.Series([model.predict([[lookahead]])])  # Predicción del modelo
        r2 = r2_score(y_true, y_pred)  # Coeficiente de determinación


        # Avoid NaN values
        if pd.isna(beta) or pd.isna(r2):
            continue
        
        momentum_series.iloc[index + lookahead] = beta
        r2_series.iloc[index + lookahead] = r2

    # Guardar las series en FileCache
    momentum_path = os.path.join(cache_path, f"model-momentum-{model_name}-{ticker}.pkl")
    r2_path = os.path.join(cache_path, f"model-momentum-{model_name}-{ticker}@r2.pkl")

    with open(momentum_path, 'wb') as file:
        pickle.dump(momentum_series, file)

    with open(r2_path, 'wb') as file:
        pickle.dump(r2_series, file)

