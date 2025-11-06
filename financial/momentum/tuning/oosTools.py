'''

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import numpy as np
import pandas as pd
from copy import deepcopy
import json
from financial.lab.evaluation import ModelEvaluator
from financial.momentum.experiment.modelExperiment import ModelExperimentFactory
from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory
from financial.momentum.utilities import send_telegram_message

def quick_oos_metrics(ds, ticker, pred_oos, lookahead, start, end):
    """
    Compute out-of-sample metrics for given predictions.
    """
    P = ds.get_data(ticker, start, end).astype(float)
    df = pd.DataFrame({"P": P, "pred": pred_oos}).dropna()
    L = lookahead

    df["y_true"] = (df["P"].shift(-L) - df["P"]) / df["P"]
    df = df.dropna()                        

    err  = df["pred"] - df["y_true"]
    mae  = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    r2   = 1 - ((err**2).sum() / ((df["y_true"] - df["y_true"].mean())**2).sum() + 1e-12)
    corr = df["pred"].corr(df["y_true"])
    hit  = (np.sign(df["pred"]) == np.sign(df["y_true"])).mean()

    return {
        "n": int(df.shape[0]),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "corr": float(corr),
        "hit_rate": float(hit),
    }


class OOSResult:
    def __init__(self, metrics_dict: dict):
        self.m = metrics_dict  # {"MAE":..., "RMSE":..., "R2":..., "corr":..., "hit_rate":...}
    def metric(self, k: str):
        # - means "minimize", no - means "maximize"
        return -float(self.m[k[1:]]) if k.startswith("-") else float(self.m[k])
    def __str__(self):
        return f"OOSResult({self.m})"

class OOSMomentumEvaluator(ModelEvaluator):
    """
    Evaluator for out-of-sample project models.
    """
    def __init__(self, ds, ticker, start, end, lookahead, horizon, quick_oos_metrics_fn, default_architecture):
        self.ds, self.ticker = ds, ticker
        self.start, self.end = start, end
        self.lookahead, self.horizon = lookahead, horizon
        self.quick = quick_oos_metrics_fn
        self.factory = KerasAdvancedModelFactory()
        self.default_architecture = default_architecture

    def evaluate_model(self, experiment_id: str, hyperparams_merged: dict):
        print("[DBG] config_selection merged model dict:", hyperparams_merged.get("model", {}))

        hp = deepcopy(hyperparams_merged)
        hp.setdefault("model", {})
        arch = hp["model"].get("architecture") or getattr(self, "default_architecture", None)
        if not arch:
            raise ValueError("No model.architecture given in hyperparams or default.")
        hp["model"]["architecture"] = arch

        print("[DBG] hp(model) ->", json.dumps(hp.get("model", {}), indent=2))

        config = {
            "mode": "global",
            "datastore": self.ds,
            "ticker": self.ticker,
            "model_factory": self.factory,
            "name": experiment_id,
            "start_year": self.start,
            "end_year":   self.end,
            "lookahead":  self.lookahead,
            "horizon":    self.horizon,
            "model_params": hp
        }

        exp = ModelExperimentFactory.create_experiment(config)
        exp.run()

        yhat_oos = pd.Series(exp.predictions, name="pred").astype(float)

        m = self.quick(self.ds, self.ticker, yhat_oos,
                       self.lookahead, self.start, self.end)
        
        self.last_oos_series = yhat_oos
        self.last_metrics    = m
        self.last_config     = hyperparams_merged
        send_telegram_message(f"Model runned: {experiment_id}")
        return (None, {"oos": OOSResult(m)})
