'''
Hyperparameters optimizator for KerasAdvanced models with ticker SP500.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory
from financial.momentum.experiment.modelExperiment import ModelExperimentFactory

# Importing utilities:
from financial.momentum.tuning.oosTools import quick_oos_metrics
from financial.momentum.utilities import find_dotenv, reset_gpu
from dotenv import load_dotenv

import os
import itertools

import pandas as pd
import tensorflow as tf

# Importing financial:
import financial.data as fd
from financial.model import Model
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy


def manual_hyper_tuning(ticker: str, params_grid: dict, name:str):
    '''
    Manual hyperparameter tuning for a specific model.
    '''
    # Loading environment variables:
    load_dotenv(find_dotenv())
    lookahead = 20
    horizon = 90
    start_date = "1990-01-01"
    tuning_end_date = "2023-12-31"
    holdout_end_date = "2024-06-30"

    datastore = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"]+"/", update_strategy=NoUpdateStrategy()))
    factory = KerasAdvancedModelFactory()
    def _run_one(params: dict, end_date: str):
        '''
        Runs one experiment with the given parameters.
        '''
        config = {
            "mode": "global",  
                "datastore": datastore,
                "ticker": ticker,
                "model_factory": factory,
                "name": name,
                "start_year": start_date,
                "end_year": end_date,
                "lookahead": lookahead,
                "horizon": horizon,
                "model_params": params,
        }

        exp = ModelExperimentFactory().create_experiment(config)
        exp.run()
        pred_oss = exp.predictions

        metrics = quick_oos_metrics(datastore, ticker, pred_oss, lookahead, start_date, end_date)

        reset_gpu()
        return pred_oss, metrics
    
    keys, values = zip(*params_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    rows = []
    results = []

    for i, params in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {params}")
        _, m = _run_one(params, tuning_end_date)
        m_row = {"config": params, **m}
        rows.append(m_row)

    df = pd.DataFrame(rows).sort_values(by=["corr", "R2"], ascending=[False, False]).reset_index(drop=True)
    print("\nTOP (tuning):")
    print(df.head(5))

    best_params = df.loc[0, "config"]

    # ========== HOLDOUT (2024–2025) ==========
    print("\nEvaluating best config on holdout (blind)…")
    pred_holdout, m_holdout = _run_one(best_params, holdout_end_date)
    print("Holdout:", m_holdout)

    return df, best_params, m_holdout, pred_holdout


def save_tuning_results(df, best_cfg, holdout_metrics, pred_holdout, ticker, name_prefix):
    '''
    Saves the tuning results to disk.
    '''
    base_path = os.environ["MODEL"] + "/tuning"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    df.to_csv(os.path.join(base_path, f"{name_prefix}_{ticker}_tuning_results.csv"), index=False)
    
    with open(os.path.join(base_path, f"{name_prefix}_{ticker}_best_config.json"), "w") as f:
        import json
        json.dump(best_cfg, f, indent=4)
    
    with open(os.path.join(base_path, f"{name_prefix}_{ticker}_holdout_metrics.json"), "w") as f:
        json.dump(holdout_metrics, f, indent=4)
    
    pd.Series(pred_holdout).to_csv(os.path.join(base_path, f"{name_prefix}_{ticker}_pred_holdout.csv"), header=["pred"], index_label="date")


if __name__ == "__main__":

    params_grid_cnn = {
    "architecture": ["cnn"],
    "topology": [
        {"layers":[64,32], "activation":{"hidden":"relu","output":"linear"}},
        {"layers":[128,64], "activation":{"hidden":"relu","output":"linear"}},
    ],
    "optimization": [
        {"optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}}
    ],
    "model": [
        # Light capacity
        {"n_blocks":2, "filters":[64,64],  "kernel_sizes":[7,3], "padding":"same",
        "pool_every":1, "pool_size":2, "dropout":0.2, "l2":1e-6, "batch_norm":True, "global_pool":True},
        # More capacity
        {"n_blocks":3, "filters":[64,64,64], "kernel_sizes":[5,3,3], "padding":"same",
        "pool_every":1, "pool_size":2, "dropout":0.3, "l2":1e-6, "batch_norm":True, "global_pool":True},
        # Longer receptive field with dilations
        {"n_blocks":3, "filters":[64,64,64], "kernel_sizes":[3,3,3], "dilations":[1,2,4], "padding":"causal",
        "pool_every":0, "dropout":0.2, "l2":1e-6, "batch_norm":True, "global_pool":True},
    ],
    }

    ticker = "^GSPC"
    name = "keras_cnn_sp500"
    df_res, best_cfg, holdout_metrics, pred_holdout = manual_hyper_tuning(
        ticker=ticker,
        params_grid=params_grid_cnn,
        name=name,
    )

    save_tuning_results(
        df_res,
        best_cfg,
        holdout_metrics,
        pred_holdout,
        ticker=ticker,
        name_prefix=name,
    )
