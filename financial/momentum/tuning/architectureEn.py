'''
Architecture engineering for the best model found in the hyperparameter tuning.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''


import os
import pandas as pd
import itertools
import financial.data as fd
from dotenv import load_dotenv
from financial.io.cache import NoUpdateStrategy
from financial.io.file.cache import FileCache

from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory
from financial.momentum.experiment.modelExperiment import ModelExperimentFactory
from financial.momentum.tuning.oosTools import quick_oos_metrics
from financial.momentum.utilities import store_results, reset_gpu, find_dotenv, send_telegram_message


def architecture_engineering( datastore, start_date, end_date, ticker="^GSPC",extra_tickers = None):

    lookahead = 20
    horizon = 90
    if extra_tickers:
            ticker_field = [ticker] + extra_tickers
    else:
        ticker_field = ticker 
    
    factory = KerasAdvancedModelFactory()
    
    if extra_tickers:
        extras_str = "_".join(extra_tickers)
        name = f"keras_transformer_{ticker}_{extras_str}_{end_date[:4]}_multiple_tuning"
    else:
        name = f"keras_transformer_{ticker}_{end_date[:4]}_multiple_tuning"


    path = os.path.join(os.environ["MODEL"], "tuning/^GSPC_architecture_engineering.csv")

    if os.path.isfile(path):
        df = pd.read_csv(path)
        if name in df["name"].values:
            print(f"Architecture engineering already done for {name}. Skipping.")
            return

    winner_architecture = {
        "mode": "global",  
            "datastore": datastore,
            "ticker": ticker_field,
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

    exp = ModelExperimentFactory.create_experiment(winner_architecture)
    exp.run()

    yhat_oos = pd.Series(exp.predictions, name="pred").astype(float)
    store_results(ticker, name, yhat_oos, exp.predictions, os.environ["CACHE"], os.environ["MODEL"])
    metrics = quick_oos_metrics(datastore, ticker, yhat_oos, lookahead, start_date, end_date)

    row = {
           "name": name,
           **{k: float(metrics[k]) for k in ["n","MAE","RMSE","R2","corr","hit_rate"]},
    }
    df = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not os.path.isfile(path), index=False)

    send_telegram_message(f"Architecture engineering finished for {name}. Metrics: {metrics}")
    reset_gpu()

if __name__ == "__main__":
    load_dotenv(dotenv_path=find_dotenv())
    datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/", update_strategy=NoUpdateStrategy()))
    start_date = "1990-01-01"
    end_date = "2025-06-30"

    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["DX-Y.NYB"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["GC=F"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["M2NS"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["^TNX"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["^VIX"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["^MOVE"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["^VIX","^MOVE"])
    # architecture_engineering(datastore, start_date, end_date, ticker="^GSPC", extra_tickers=["M2NS","^TNX"])
    
    tickers = ["DX-Y.NYB", "GC=F", "M2NS", "^TNX", "^VIX", "^MOVE", "PAYEMS", "DPRIME"]

    for r in range(1, len(tickers) + 1):
        for combo in itertools.combinations(tickers, r):
            architecture_engineering(
                datastore, start_date, end_date,
                ticker="^GSPC", extra_tickers=list(combo)
            )
