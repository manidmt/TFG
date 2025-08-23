'''
Model Creator file. This file is responsible for creating and managing models in the financial momentum project.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''


# Importing factories:
from financial.lab import experiment
from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory
from financial.momentum.models.randomForest import RandomForestModelFactory
from financial.momentum.models.SVR import SVRModelFactory
from financial.momentum.models.exponentialRegression import ExponentialRegressionModelFactory
from financial.momentum.experiment.modelExperiment import ModelExperimentFactory

# Importing utilities:
from financial.momentum.utilities import find_dotenv, metrics, send_telegram_message, reset_gpu, store_results

import os
from dotenv import load_dotenv

# Importing financial:
import financial.data as fd
from financial.model import Model
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy


def create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers = None):
    """
    
    """
    load_dotenv(dotenv_path=find_dotenv())
    #os.environ["MODEL"] = "/home/manidmt/TFG/OTRI/models/keras/"

    data = datastore.get_data(ticker, start_date, end_date)
    target = data[lookahead + horizon:]
    factory = KerasAdvancedModelFactory()
    architectures = ["rnn","lstm", "cnn", "cnn2d", "transformer"]
    layers = [64, 32]
    activation = {"hidden": "relu", "output": "linear"}
    batch_size = 16
    epochs = 10

    for arch in architectures:
        if extra_tickers == None:
            name = f"keras_{arch}_{ticker}_{end_date[:4]}_single"
        else:
            for extra in extra_tickers:
                name = f"keras_{arch}_{ticker}_{extra}_{end_date[:4]}_multiple"

        try:
            if Model.from_file(name, path=os.environ["MODEL"]):
                print(f"Modelo {name} ya existe. Saltando...")
                #send_telegram_message(f"Modelo {name} ya existe. Saltando...")
                continue
        except FileNotFoundError:
            print(f"Modelo {name} no existe. Procediendo a entrenar...")
            send_telegram_message(f"Modelo {name} no existe. Procediendo a entrenar...")

        if extra_tickers:
            ticker_field = [ticker] + extra_tickers
        else:
            ticker_field = ticker 

        config = {
            "mode": "global",  
            "datastore": datastore,
            "ticker": ticker_field,
            "model_factory": factory,
            "name": name,
            "start_year": start_date,
            "end_year": end_date,
            "lookahead": 20,
            "horizon": 90,
            "model_params": {
                "architecture": arch,
                "topology": {
                    "layers": layers,
                    "activation": {
                        "hidden": activation["hidden"],
                        "output": activation["output"]
                    }
                },
                "optimization": {
                    "optimizer": "adam",
                    "loss": "mean_squared_error",
                    "metrics": ["mae"],
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "validation_split": 0.1
                }
            }
        }
        try:
            experiment = ModelExperimentFactory.create_experiment(config)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, os.environ["CACHE"], os.environ["MODEL"])
            send_telegram_message("Resultados guardados")
            metrics(experiment, preds, target, os.environ["MODEL"])
            send_telegram_message("Metricas guardadas")
            reset_gpu()
        except Exception as e:
            print(f"Falló el modelo {name}: {e}")
            send_telegram_message(f"Falló el modelo {name}: {e}")


def create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon):
    """
    
    """
    load_dotenv(dotenv_path=find_dotenv())
    #os.environ["MODEL"] = "/home/manidmt/TFG/OTRI/models/scikit-learn/"

    data = datastore.get_data(ticker, start_date, end_date)
    target = data[lookahead + horizon:]

    # Clenow
    name = f"scikit-learn_clenow_{ticker}_{end_date[:4]}"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Modelo {name} ya existe. Saltando...")
            # send_telegram_message(f"Modelo {name} ya existe. Saltando...")
    except FileNotFoundError:
        print(f"Modelo {name} no existe. Procediendo a entrenar...")
        send_telegram_message(f"Modelo {name} no existe. Procediendo a entrenar...")
        factory = ExponentialRegressionModelFactory()
        config = {
            "mode": "local",  
            "datastore": datastore,
            "ticker": ticker,
            "model_factory": factory,
            "name": name,
            "start_year": start_date,
            "end_year": end_date,
            "lookahead": 20,
            "horizon": 90,
        }
        try:
            experiment = ModelExperimentFactory.create_experiment(config)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, os.environ["CACHE"], os.environ["MODEL"])
            send_telegram_message("Resultados guardados")
            metrics(experiment, preds, target, os.environ["MODEL"], global_model=False)
            send_telegram_message("Metricas guardadas")
        except Exception as e:
            print(f"Falló el modelo {name}: {e}")
            send_telegram_message(f"Falló el modelo {name}: {e}")
        
    # SVR
    name = f"scikit-learn_svr_{ticker}_{end_date[:4]}"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Modelo {name} ya existe. Saltando...")
            # send_telegram_message(f"Modelo {name} ya existe. Saltando...")
    except FileNotFoundError:
        print(f"Modelo {name} no existe. Procediendo a entrenar...")
        send_telegram_message(f"Modelo {name} no existe. Procediendo a entrenar...")
        factory = SVRModelFactory()
        config = {
            "mode": "global",
            "datastore": datastore,
            "ticker": ticker,
            "model_factory": factory,
            "name": name,
            "start_year": start_date,
            "end_year": end_date,
            "lookahead": 20,
            "horizon": 90,
        }
        try:
            experiment = ModelExperimentFactory.create_experiment(config)
            print(experiment)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, os.environ["CACHE"], os.environ["MODEL"])
            send_telegram_message("Resultados guardados")
            metrics(experiment, preds, target, os.environ["MODEL"])
            send_telegram_message("Metricas guardadas")
        except Exception as e:
            print(f"Falló el modelo {name}: {e}")
            send_telegram_message(f"Falló el modelo {name}: {e}")
    
    # RandomForest
    name = f"scikit-learn_randomforest_{ticker}_{end_date[:4]}"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Modelo {name} ya existe. Saltando...")
            # send_telegram_message(f"Modelo {name} ya existe. Saltando...")
    except FileNotFoundError:
        print(f"Modelo {name} no existe. Procediendo a entrenar...")
        send_telegram_message(f"Modelo {name} no existe. Procediendo a entrenar...")
        factory = RandomForestModelFactory()
        config = {
            "mode": "global",
            "datastore": datastore,
            "ticker": ticker,
            "model_factory": factory,
            "name": name,
            "start_year": start_date,
            "end_year": end_date,
            "lookahead": 20,
            "horizon": 90,
        }
        try:
            print(f"config: {config}")
            experiment = ModelExperimentFactory.create_experiment(config)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, os.environ["CACHE"], os.environ["MODEL"])
            send_telegram_message("Resultados guardados")
            metrics(experiment, preds, target, os.environ["MODEL"])
            send_telegram_message("Metricas guardadas")
        except Exception as e:
            print(f"Falló el modelo {name}: {e}")
            send_telegram_message(f"Falló el modelo {name}: {e}")


if __name__ == "__main__":
    print("Initializing Model Creation")
    load_dotenv(dotenv_path=find_dotenv())
    datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/", update_strategy=NoUpdateStrategy()))
    start_date = "1990-01-01"
    end_date = "2025-06-30"
    lookahead = 20
    horizon = 90

    # ETFs
    tickers = ["QQQ", "SPY", "URTH", "IYY", "EIMI"]
    print(os.environ["MODEL"])
    for ticker in tickers:
       if os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/keras":
           #create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon)
           create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers=["M2NS"])
       elif os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/scikit-learn":
           create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon)

    # Top Tech
    tickers = ["AAPL", "GOOG", "TSLA", "MSFT", "NVDA", "AMZN", "META", "BAM", "INTC", "QCOM", "ASML", "ACN", "ORCL", "NVS", "UNH"]
    for ticker in tickers:
        if os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/keras":
            create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon)
            create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers=["M2NS"])
        elif os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/scikit-learn":
            create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon)
