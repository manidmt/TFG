'''

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''


import os, json
import pandas as pd

from dotenv import load_dotenv
import financial.data as fd
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy

from financial.momentum.tuning.oosTools import quick_oos_metrics
from financial.momentum.tuning.oosTools import OOSMomentumEvaluator
from financial.momentum.tuning.hyperSpaces import build_cnn_space


from financial.lab.tuning.optimizers import BayesianOptimizer
from financial.lab.tuning.estimators import GaussianProcess, GaussianProcessOptimizer

class OOSBayesGPO(GaussianProcessOptimizer):
    """
    Adaptador para usar el BayesianOptimizer estándar con evaluaciones 'tipo Fernando'
    (evaluation[output].metric(...)). Convertimos esas métricas en un escalar para el GP.
    """
    def __init__(self, space, gpr: GaussianProcess, restarts: int = 50, weights: dict | None = None):
        super().__init__(space, gpr, restarts)
        # pesos sobre métricas; por defecto prioriza corr y R2, penaliza MAE/RMSE
        self.weights = weights or {"corr": 1.0, "R2": 0.5, "-MAE": 0.25, "-RMSE": 0.25, "hit_rate": 0.0}

    def hyperparameter_score(self, evaluation):
        """
        'evaluation' aquí es el dict que devolvió tu evaluator:
            {"oos": OOSResult(...)}
        Sumamos ponderadamente métricas con .metric(nombre) → mayor es mejor.
        OJO: devolvemos el negativo (minimización en el GP).
        """
        oos = evaluation["oos"]
        s = 0.0
        for k, w in self.weights.items():
            s += w * float(oos.metric(k))
        return -s  # el base class minimiza



if __name__ == "__main__":
    load_dotenv()

    # ---------- Config ----------
    ticker     = "^GSPC"
    start_date = "1990-01-01"
    end_date   = "2025-06-30"
    lookahead  = 20
    horizon    = 90
    trials     = 20

    out_dir = os.path.join(os.environ["MODEL"], "tuning")
    os.makedirs(out_dir, exist_ok=True)

    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )

    # ---------- Espacio + Evaluador ----------
    space = build_cnn_space()
    evaluator = OOSMomentumEvaluator(
        ds=ds, ticker=ticker,
        start=start_date, end=end_date,
        lookahead=lookahead, horizon=horizon,
        quick_oos_metrics_fn=quick_oos_metrics
    )

    # ---------- Gaussian Process + Adaptador ----------
    gpr       = GaussianProcess()  # Matern(nu=2.5), alpha=1e-4 por defecto
    estimator = OOSBayesGPO(
        space, gpr, restarts=50,
        weights={"corr":1.0, "R2":0.5, "-MAE":0.25, "-RMSE":0.25, "hit_rate":0.0}
    )

    bayes = BayesianOptimizer(space, ds, None, evaluator, estimator=estimator, trials=trials)

    # Métricas que mostrará/seguirá el optimizador para "best"
    bayes.metrics = ["corr", "R2", "-MAE", "-RMSE", "hit_rate"]
    bayes.best = {
        m: {"value": float("-inf"), "best_configuration": None, "best_model": None}
        for m in bayes.metrics
    }

    # ---------- Ejecutar ----------
    bayes.run()

    # ---------- Guardar TODOS los trials ----------
    rows = []
    for (cfg_sel, evaluation) in bayes.trials:
        # cfg_sel: dict de etiquetas del espacio (selección)
        metrics = evaluation["oos"].m            # dict con MAE, RMSE, R2, corr, hit_rate
        merged  = space.parameters(cfg_sel)      # hiperparámetros expandidos
        rows.append({
            "config_selection": json.dumps(cfg_sel),
            "config_merged":    json.dumps(merged),
            **metrics
        })

    df = pd.DataFrame(rows)
    trials_path = os.path.join(out_dir, f"bayes_{ticker}_trials_1990_2025.csv")
    df.to_csv(trials_path, index=False)
    print(f"[OK] Trials guardados → {trials_path}")

    # ---------- Guardar BEST por corr y por R2 ----------
    def save_best(metric_name: str):
        best_block = bayes.best[metric_name]  # {"value":..., "best_configuration":..., "best_model": None}
        best_sel   = best_block["best_configuration"]
        best_val   = best_block["value"]
        best_merged = space.parameters(best_sel)

        best_path = os.path.join(out_dir, f"bayes_{ticker}_best_{metric_name}_1990_2025.json")
        with open(best_path, "w") as f:
            json.dump({
                "metric": metric_name,
                "value": best_val,
                "config_selection": best_sel,
                "config_merged": best_merged
            }, f, indent=2)
        print(f"[OK] Best ({metric_name}) guardado → {best_path}")

    save_best("corr")
    save_best("R2")

    # ---------- Resumen por consola ----------
    print(f"\n{bayes.total_trials()} TRIALS")
    for (cfg_sel, evaluation) in bayes.trials:
        res = evaluation["oos"]
        print(cfg_sel, "→ corr=", res.metric("corr"), "R2=", res.metric("R2"))
    print("\nBEST by corr:", bayes.best["corr"])
    print("BEST by R2:  ", bayes.best["R2"])