'''
Creating resumable hyperparameter tuning optimizers (random and Bayesian) that log results to CSV files.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import os, json, traceback
import pandas as pd

from financial.lab.tuning.optimizers import RandomSearch, BayesianOptimizer
from financial.lab.tuning.space import HyperparameterSearchSpace
from financial.lab.tuning.estimators import GaussianProcess, GaussianProcessOptimizer


def _config_id(space: HyperparameterSearchSpace, selection: dict) -> str:
    return space.configuration_id(selection)

def _safe_append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    header = not os.path.isfile(path)
    df.to_csv(path, mode="a", header=header, index=False)

def _try_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _rebuild_oos_result(row: pd.Series):
    class OOSResult:
        def __init__(self, metrics_dict: dict): self.m = metrics_dict
        def metric(self, k: str):
            return -float(self.m[k[1:]]) if k.startswith("-") else float(self.m[k])
        def __str__(self): return f"OOSResult({self.m})"
    m = {
        "n":        int(row.get("n", 0)),
        "MAE":      float(row.get("MAE", 0.0)),
        "RMSE":     float(row.get("RMSE", 0.0)),
        "R2":       float(row.get("R2", 0.0)),
        "corr":     float(row.get("corr", 0.0)),
        "hit_rate": float(row.get("hit_rate", 0.0)),
    }
    return {"oos": OOSResult(m)}

def _bootstrap_from_csv(self, space, csv_path: str):
    self.done_ids = set()
    if not os.path.isfile(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    self.trials = []

    for _, row in df.iterrows():
        sel_json = row.get("config_selection")
        if not isinstance(sel_json, str):
            continue
        sel = _try_json_load(sel_json)
        if not isinstance(sel, dict):
            continue

        cfg_id = _config_id(space, sel)
        self.done_ids.add(cfg_id)

        evaluation = _rebuild_oos_result(row)  # {"oos": OOSResult(...)}
        self.trials.append((sel, evaluation))

        for metric in self.metrics:
            for output in evaluation:
                try:
                    val = evaluation[output].metric(metric)
                except KeyError:
                    continue
                if val > self.best[metric]["value"]:
                    self.best[metric] = {
                        "value": val,
                        "best_configuration": sel,
                        "best_model": None
                    }

# ---- Random resumible --------------------------------------------------------

class ResumableRandomSearch(RandomSearch):
    def __init__(self, space, ds, factory, evaluator, trials, csv_path, err_path, metrics=None):
        super().__init__(space, ds, factory, evaluator, trials)
        self.csv_path = csv_path
        self.err_path = err_path

        if metrics is not None:
            self.metrics = metrics

        self.best = {
            m: {"value": float("-inf"), "best_configuration": None, "best_model": None}
            for m in self.metrics
        }

        _bootstrap_from_csv(self, space, csv_path)

    def next(self) -> dict:
        for _ in range(500):
            sel = self.space.random()
            cid = _config_id(self.space, sel)
            if cid not in getattr(self, "done_ids", set()):
                return sel
        raise RuntimeError("No combinations left")

    def trial(self, configuration: dict) -> None:
        cfg_id = _config_id(self.space, configuration)
        try:
            trial_id  = cfg_id
            params    = self.space.parameters(configuration)
            (model, evaluation) = self.evaluator.evaluate_model(trial_id, params)

            oos = evaluation["oos"]
            row = {
                "config_id":        trial_id,
                "config_selection": json.dumps(configuration, sort_keys=True),
                "config_merged":    json.dumps(params,        sort_keys=True),
                **{k: float(oos.m[k]) for k in ["n","MAE","RMSE","R2","corr","hit_rate"]},
            }
            _safe_append_csv(self.csv_path, row)

            if not hasattr(self, "done_ids"): self.done_ids = set()
            self.done_ids.add(cfg_id)

            self.trials.append((configuration, evaluation))
            for metric in self.metrics:
                for output in evaluation:
                    val = evaluation[output].metric(metric)
                    if val > self.best[metric]["value"]:
                        self.best[metric] = {
                            "value": val,
                            "best_configuration": configuration,
                            "best_model": model
                        }

        except Exception as e:
            erow = {
                "config_id":        cfg_id,
                "config_selection": json.dumps(configuration, sort_keys=True),
                "error":            repr(e),
                "traceback":        traceback.format_exc(),
            }
            _safe_append_csv(self.err_path, erow)
            print(f"[WARN] Trial failed ({cfg_id}). Continuing…")


# ---- Bayesian resumible ------------------------------------------------------

class ResumableBayesianOptimizer(BayesianOptimizer):
    def __init__(self, space, ds, factory, evaluator,
                 estimator, trials, csv_path, err_path,
                 metrics=None,
                 initial_points=None, initial_points_per_dimension=3):
        super().__init__(space, ds, factory, evaluator,
                         estimator=estimator,
                         trials=trials,
                         initial_points=initial_points,
                         initial_points_per_dimension=initial_points_per_dimension)
        self.csv_path = csv_path
        self.err_path = err_path

        if metrics is not None:
            self.metrics = metrics

        self.best = {
            m: {"value": float("-inf"), "best_configuration": None, "best_model": None}
            for m in self.metrics
        }

        _bootstrap_from_csv(self, space, csv_path)

    def minimum_initial_points(self):
        return super().minimum_initial_points()

    def next(self) -> dict:
        """
        Igual que la lógica del padre, pero saltando configs repetidas.
        """
        # 1) inicial: puntos aleatorios
        if self.total_trials() < self.minimum_initial_points():
            for _ in range(500):
                sel = self.space.random()
                cid = _config_id(self.space, sel)
                if cid not in getattr(self, "done_ids", set()):
                    return sel
            raise RuntimeError("No quedan combinaciones nuevas para la fase inicial.")

        # 2) explotación: propuestas del GP (evitando duplicados)
        for _ in range(500):
            sel = self.estimator.propose(self.trials)  # devuelve un dict de selección
            # Si el estimator devuelve un "params merged", lo convertimos a selección;
            # En nuestro caso, propone selección (claves del space)
            cid = _config_id(self.space, sel)
            if cid not in getattr(self, "done_ids", set()):
                return sel

        # 3) fallback: aleatorio sin repetir
        for _ in range(500):
            sel = self.space.random()
            cid = _config_id(self.space, sel)
            if cid not in getattr(self, "done_ids", set()):
                return sel

        raise RuntimeError("No combinations left")

    def trial(self, configuration: dict) -> None:
        cfg_id = _config_id(self.space, configuration)
        try:
            trial_id  = cfg_id
            params    = self.space.parameters(configuration)
            (model, evaluation) = self.evaluator.evaluate_model(trial_id, params)

            oos = evaluation["oos"]
            row = {
                "config_id":        trial_id,
                "config_selection": json.dumps(configuration, sort_keys=True),
                "config_merged":    json.dumps(params,        sort_keys=True),
                **{k: float(oos.m[k]) for k in ["n","MAE","RMSE","R2","corr","hit_rate"]},
            }
            _safe_append_csv(self.csv_path, row)

            if not hasattr(self, "done_ids"): self.done_ids = set()
            self.done_ids.add(cfg_id)

            self.trials.append((configuration, evaluation))
            for metric in self.metrics:
                for output in evaluation:
                    val = evaluation[output].metric(metric)
                    if val > self.best[metric]["value"]:
                        self.best[metric] = {
                            "value": val,
                            "best_configuration": configuration,
                            "best_model": model
                        }

        except Exception as e:
            erow = {
                "config_id":        cfg_id,
                "config_selection": json.dumps(configuration, sort_keys=True),
                "error":            repr(e),
                "traceback":        traceback.format_exc(),
            }
            _safe_append_csv(self.err_path, erow)
            print(f"[WARN] Trial failed ({cfg_id}). Continuing…")



from dotenv import load_dotenv
import os
import sys
import financial.data as fd
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy

from financial.momentum.tuning.hyperSpaces import build_cnn_space, build_lstm_space, build_transformer_space
from financial.momentum.tuning.oosTools import OOSMomentumEvaluator, quick_oos_metrics
from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory

from financial.lab.tuning.estimators import GaussianProcess
from financial.momentum.tuning.bayesTuning import OOSBayesGPO
from financial.lab.tuning.optimizers import BayesianOptimizer


if __name__ == "__main__":
    load_dotenv()
    ds = fd.CachedDataStore(path=os.environ["DATA"],
                            cache=FileCache(cache_path=os.environ["CACHE"]+"/", update_strategy=NoUpdateStrategy()))
    
    architecture = sys.argv[1] if len(sys.argv) > 1 else "transformer"

    if architecture == "cnn":
        space = build_cnn_space()
    elif architecture == "lstm":
        space = build_lstm_space()
    elif architecture == "transformer":
        space = build_transformer_space()
    else:
        space = build_transformer_space()


    evaltr  = OOSMomentumEvaluator(ds, "^GSPC", "1990-01-01", "2024-12-31", lookahead=20, horizon=90,
                                quick_oos_metrics_fn=quick_oos_metrics, default_architecture=architecture)
    
    method = sys.argv[2] if len(sys.argv) > 2 else "random"

    out_dir = os.path.join(os.environ["MODEL"], "tuning")
    csv     = os.path.join(out_dir, f"{method}_^GSPC_trials_{architecture}.csv")
    errcsv  = os.path.join(out_dir, f"{method}_^GSPC_errors_{architecture}.csv")

    if method == "random":
        search = ResumableRandomSearch(space, ds, KerasAdvancedModelFactory(), evaltr, trials=15, metrics=["corr", "R2", "-MAE", "-RMSE", "hit_rate"], csv_path=csv, err_path=errcsv)
    elif method == "bayes":
        gpr = GaussianProcess(
        alpha=1e-3,             
        beta=3.5,              
        seed=42,
        optimizer_restarts=12
        )

        estimator = OOSBayesGPO(
            space, gpr,
            restarts=50,
            weights={"corr":1.0, "R2":0.25, "-MAE":0.25, "-RMSE":0.25, "hit_rate":0.2}
        )
        # initial_pts = max(18, 3*space.dimensions())
        search = ResumableBayesianOptimizer(space, ds, KerasAdvancedModelFactory(), evaltr, estimator, trials=15, initial_points=0, metrics=["corr", "R2", "-MAE", "-RMSE", "hit_rate"], csv_path=csv, err_path=errcsv)

    search.metrics = ["corr","R2","-MAE","-RMSE","hit_rate"]
    print("Dir for output:", out_dir)
    print("Trials CSV:", csv)
    print("Errors CSV:", errcsv)
    print("Search method:", method)
    print("Architecture:", architecture)
    search.run()

    print("BEST corr:", search.best["corr"])
    print("BEST R2:",   search.best["R2"])


