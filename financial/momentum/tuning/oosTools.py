'''

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import numpy as np
import pandas as pd
from financial.lab.evaluation import ModelEvaluator
from financial.momentum.experiment.modelExperiment import ModelExperimentFactory
from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory


def quick_oos_metrics(ds, ticker, pred_oos, lookahead, start, end):
    """
    pred_oos: Serie OOS (ŷ_t) que predice el retorno relativo futuro a L días vista en t.
    Métricas OOS contra y_t = (P_{t+L}-P_t)/P_t, sin reconstruir precios.
    """
    P = ds.get_data(ticker, start, end).astype(float)
    df = pd.DataFrame({"P": P, "pred": pred_oos}).dropna()
    L = lookahead

    # y_true = retorno futuro
    df["y_true"] = (df["P"].shift(-L) - df["P"]) / df["P"]
    df = df.dropna()                        # alinea con pred

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


# Objeto compatible con .metric() que espera el optimizador de Fernando
class OOSResult:
    def __init__(self, metrics_dict: dict):
        self.m = metrics_dict  # {"MAE":..., "RMSE":..., "R2":..., "corr":..., "hit_rate":...}
    def metric(self, k: str):
        # Si el nombre empieza por "-", lo tratamos como métrica a minimizar
        return -float(self.m[k[1:]]) if k.startswith("-") else float(self.m[k])
    def __str__(self):
        return f"OOSResult({self.m})"

class OOSMomentumEvaluator(ModelEvaluator):
    """
    Entrena con tu GlobalModelExperiment y devuelve métricas OOS
    sobre la serie concatenada (sin fugas temporales).
    """
    def __init__(self, ds, ticker, start, end, lookahead, horizon, quick_oos_metrics_fn):
        self.ds, self.ticker = ds, ticker
        self.start, self.end = start, end
        self.lookahead, self.horizon = lookahead, horizon
        self.quick = quick_oos_metrics_fn
        self.factory = KerasAdvancedModelFactory()

    def evaluate_model(self, experiment_id: str, hyperparams_merged: dict):
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
            "model_params": hyperparams_merged,
        }

        exp = ModelExperimentFactory.create_experiment(config)
        exp.run()

        # Tu Walk-Forward ya deja exp.predictions = serie OOS concatenada
        yhat_oos = pd.Series(exp.predictions, name="pred").astype(float)

        m = self.quick(self.ds, self.ticker, yhat_oos,
                       self.lookahead, self.start, self.end)
        
        self.last_oos_series = yhat_oos
        self.last_metrics    = m
        self.last_config     = hyperparams_merged

        # Devuelve (model, evaluation). El optimizador usa evaluation["oos"].metric(...)
        return (None, {"oos": OOSResult(m)})