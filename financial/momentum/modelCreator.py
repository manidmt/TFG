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
from financial.momentum.tuning.oosTools import quick_oos_metrics 

import os
import pandas as pd
from dotenv import load_dotenv

# Importing financial:
import financial.data as fd
from financial.model import Model
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy


def save_GSPC(given_metrics, name):
    """
    Save the metrics for the GSPC model.
    """
    path = os.path.join(os.environ["MODEL"], "^GSPC_all_models_metrics.csv")

    if os.path.isfile(path):
        df = pd.read_csv(path)
    
    row = {
           "name": name,
           **{k: float(given_metrics[k]) for k in ["n","MAE","RMSE","R2","corr","hit_rate"]},
    }
    df = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not os.path.isfile(path), index=False)

def create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers = None):
    """
    Create a Keras model for the given ticker.
    """
    load_dotenv(dotenv_path=find_dotenv())
    #os.environ["MODEL"] = "/home/manidmt/TFG/OTRI/models/keras/"

    data = datastore.get_data(ticker, start_date, end_date)
    target = data[lookahead + horizon:]
    factory = KerasAdvancedModelFactory()
    architectures = ["cnn","lstm", "rnn", "transformer"]
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
                print(f"Model {name} already exists. Skipping...")
                ##send_telegram_message(f"Model {name} already exists. Skipping...")
                continue
        except FileNotFoundError:
            print(f"Model {name} does not exist. Training...")
            #send_telegram_message(f"Model {name} does not exist. Training...")

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
            store_results(ticker, name, preds, experiment.predictions, os.environ["CACHE"], os.environ["MODEL"])
            #send_telegram_message("Results saved")
            metrics(experiment, preds, target, os.environ["MODEL"])

            if ticker == "^GSPC":
                yhat_oos = pd.Series(experiment.predictions, name="pred").astype(float)
                gMetrics = quick_oos_metrics(datastore, ticker, yhat_oos, lookahead, start_date, end_date)
                save_GSPC(gMetrics, name)

            #send_telegram_message("Metrics saved")
            reset_gpu()
        except Exception as e:
            print(f"Model {name} failed: {e}")
            #send_telegram_message(f"Model {name} failed: {e}")


def create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon):
    """
    Create a Scikit-Learn model for the given ticker.
    """
    load_dotenv(dotenv_path=find_dotenv())
    #os.environ["MODEL"] = "/home/manidmt/TFG/OTRI/models/scikit-learn/"

    data = datastore.get_data(ticker, start_date, end_date)
    target = data[lookahead + horizon:]

    # Clenow
    name = f"scikit-learn_clenow_{ticker}_{end_date[:4]}"
    if os.path.exists(f"{os.environ['MODEL']}/{name}_metrics.json"):
        print(f"Model {name} exists. Skipping...")
            # send_telegram_message(f"Modelo {name} ya existe. Saltando...")
    else:
        print(f"Model {name} does not exist. Training...")
        send_telegram_message(f"Model {name} does not exist. Training...")
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
            store_results(ticker, name, preds, experiment.predictions, os.environ["CACHE"], os.environ["MODEL"])
            #send_telegram_message("Results saved")
            metrics(experiment, preds, target, os.environ["MODEL"], global_model=False)

            if ticker == "^GSPC":
                yhat_oos = pd.Series(experiment.predictions, name="pred").astype(float)
                gMetrics = quick_oos_metrics(datastore, ticker, yhat_oos, lookahead, start_date, end_date)
                save_GSPC(gMetrics, name)

            #send_telegram_message("Metrics saved")
        except Exception as e:
            print(f"Model {name} failed: {e}")
            #send_telegram_message(f"Model {name} failed: {e}")

    # SVR
    name = f"scikit-learn_svr_{ticker}_{end_date[:4]}"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Model {name} exists. Skipping...")
            # send_telegram_message(f"Model {name} exists. Skipping...")
    except FileNotFoundError:
        print(f"Model {name} does not exist. Training...")
        #send_telegram_message(f"Model {name} does not exist. Training...")
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
            store_results(ticker, name, preds, experiment.predictions, os.environ["CACHE"], os.environ["MODEL"])
            ##send_telegram_message("Results saved")
            metrics(experiment, preds, target, os.environ["MODEL"])

            if ticker == "^GSPC":
                yhat_oos = pd.Series(experiment.predictions, name="pred").astype(float)
                gMetrics = quick_oos_metrics(datastore, ticker, yhat_oos, lookahead, start_date, end_date)
                save_GSPC(gMetrics, name)

            ##send_telegram_message("Metrics saved")
        except Exception as e:
            print(f"Model {name} failed: {e}")
            ##send_telegram_message(f"Model {name} failed: {e}")

    # RandomForest
    name = f"scikit-learn_randomforest_{ticker}_{end_date[:4]}"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Model {name} exists. Skipping...")
            # #send_telegram_message(f"Model {name} exists. Skipping...")
    except FileNotFoundError:
        print(f"Model {name} does not exist. Training...")
        #send_telegram_message(f"Model {name} does not exist. Training...")
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
            experiment = ModelExperimentFactory.create_experiment(config)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, experiment.predictions, os.environ["CACHE"], os.environ["MODEL"])
            #send_telegram_message("Results saved")
            metrics(experiment, preds, target, os.environ["MODEL"])
            if ticker == "^GSPC":
                yhat_oos = pd.Series(experiment.predictions, name="pred").astype(float)
                gMetrics = quick_oos_metrics(datastore, ticker, yhat_oos, lookahead, start_date, end_date)
                save_GSPC(gMetrics, name)

            #send_telegram_message("Metrics saved")
        except Exception as e:
            print(f"Model {name} failed: {e}")
            #send_telegram_message(f"Model {name} failed: {e}")

def create_best_model(ticker, datastore, start_date, end_date, lookahead, horizon):
    """
    Create the best model (after hyperparameter tuning and feature engineering) for the given ticker.
    """
    best_feature = "^VIX"
    name = f"keras_transformer_{ticker}_{best_feature}_{end_date[:4]}_multiple"
    try:
        if Model.from_file(name, path=os.environ["MODEL"]):
            print(f"Model {name} exists. Skipping...")
            # send_telegram_message(f"Model {name} exists. Skipping...")
    except FileNotFoundError:
        print(f"Model {name} does not exist. Training...")
        send_telegram_message(f"Model {name} does not exist. Training...")
        data = datastore.get_data(ticker, start_date, end_date)
        target = data[lookahead + horizon:]
        factory = KerasAdvancedModelFactory()
        winner_architecture = {
            "mode": "global",  
                "datastore": datastore,
                "ticker": [ticker, best_feature],
                "model_factory": factory,
                "name": name,
                "start_year": start_date,
                "end_year": end_date,
                "lookahead": lookahead,
                "horizon": horizon,
                "model_params": {
                    "architecture": "transformer",
                    "dropout": 0.1,
                    "ff_dim": 64,
                    "num_heads": 4,
                    "optimization": {
                        "batch_size": 32,
                        "callbacks": {
                            "early_stopping": {"patience": 8},
                            "reduce_on_plateau": {"factor": 0.5, "patience": 4}
                        },
                        "epochs": 80,
                        "loss": "huber",
                        "optimizer": "adam",
                        "validation_split": 0.1
                    },
                    "topology": {
                        "activation": {"hidden": "relu", "output": "linear"},
                        "layers": [64, 32]
                    }
                }
            }
        try:
            experiment = ModelExperimentFactory.create_experiment(winner_architecture)
            experiment.run()
            preds = experiment.reconstruct_absolute_predictions_from_relative()
            store_results(ticker, name, preds, experiment.predictions, os.environ["CACHE"], os.environ["MODEL"])
            #send_telegram_message("Results saved")
            metrics(experiment, preds, target, os.environ["MODEL"])
            #send_telegram_message("Metrics saved")
        except Exception as e:
            print(f"Model {name} failed: {e}")
            send_telegram_message(f"Model {name} failed: {e}")

if __name__ == "__main__":
    print("Initializing Model Creation")
    load_dotenv(dotenv_path=find_dotenv())
    datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/", update_strategy=NoUpdateStrategy()))
    start_date = "1990-01-01"
    end_date = "2025-06-30"
    lookahead = 20
    horizon = 90

    # Top
    tickers = ["CVX", "PFE", "JNJ",
               "DIS", "WBD", "XOM",
               "WFC", "BRK-B", "UNH",
               "AAPL", "MSFT", "BAC",  
               "KO", "MCD", "GM", "F"]
    for ticker in tickers:
       if os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/keras":
           create_best_model(ticker, datastore, start_date, end_date, lookahead, horizon)
            #create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon)
            #create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers=["M2NS"])
            #create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers=["^IXIC"])
       elif os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/scikit-learn":
          create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon)

    # ETFs
    tickers = [
        "CSPX",   # S&P 500
        "IWDA",   # MSCI World
        "EEM",    # Emerging Markets

        "QQQ",    # Nasdaq 100
        "XLF",    # Financials
        "XLE",    # Energy
        "XLV",    # Healthcare
        "XLP",    # Consumer Staples
        "XLI",    # Industrials

        "GLD",    # Gold
        "DBC",    # Broad Commodities

        "SHY",    # 1-3 yr Treasury
        "IEF",    # 7-10 yr Treasury
        "TIP",    # TIPS / inflación
        "AGG",    # Aggregate bonds (añadido)

        "RSP",    # Equal Weight S&P 500
        "USMV",   # Minimum Volatility (añadido)
        "VYM",    # High Dividend Yield (añadido)

        "FEZ",    # Euro Stoxx 50
        "EWJ"     # MSCI Japan
    ]
    # print(os.environ["MODEL"])
    for ticker in tickers:
       if os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/keras":
           create_best_model(ticker, datastore, start_date, end_date, lookahead, horizon)
    #        create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon)
    #        #create_keras_model(ticker, datastore, start_date, end_date, lookahead, horizon, extra_tickers=["M2NS"])
       elif os.environ["MODEL"] == "/home/manidmt/TFG/OTRI/models/scikit-learn":
           create_sklearn_model(ticker, datastore, start_date, end_date, lookahead, horizon)

    
