'''

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

# run_random.py
import os, json
import pandas as pd
from dotenv import load_dotenv
import financial.data as fd
from financial.io.file.cache import FileCache
from financial.io.cache import NoUpdateStrategy
from financial.lab.tuning.optimizers import RandomSearch
from financial.momentum.models.kerasAdvanced import KerasAdvancedModelFactory

from financial.momentum.tuning.oosTools import quick_oos_metrics
from financial.momentum.tuning.oosTools import OOSMomentumEvaluator
from financial.momentum.tuning.hyperSpaces import build_transformer_space


if __name__ == "__main__":
    load_dotenv()

    # --------- Parámetros ----------
    ticker     = "^GSPC"
    start_date = "1990-01-01"
    end_date   = "2025-06-30"   # hasta mediados de 2025
    lookahead  = 20
    horizon    = 90
    trials     = 20

    out_dir = os.path.join(os.environ["MODEL"], "tuning")
    os.makedirs(out_dir, exist_ok=True)

    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )

    # --------- Espacio + Evaluador ----------
    space = build_transformer_space()
    evaluator = OOSMomentumEvaluator(
        ds=ds, ticker=ticker,
        start=start_date, end=end_date,
        lookahead=lookahead, horizon=horizon,
        quick_oos_metrics_fn=quick_oos_metrics
    )

    factory = KerasAdvancedModelFactory()
    search = RandomSearch(space, ds, factory, evaluator, trials=trials)
    # Métricas usadas por el RandomSearch para decidir "best"
    search.metrics = ["corr", "R2", "-MAE", "-RMSE", "hit_rate"]
    search.best = {
        m: {"value": float("-inf"), "best_configuration": None, "best_model": None}
        for m in search.metrics
    }
    # --------- Ejecutar ----------
    search.run()

    # --------- Guardar TODOS los trials ----------
    rows = []
    for (cfg_sel, evaluation) in search.trials:
        # cfg_sel: selección de etiquetas del espacio
        # evaluation["oos"] es OOSResult; sus métricas están en .m (dict)
        metrics = evaluation["oos"].m
        merged  = space.parameters(cfg_sel)  # hiperparámetros expandidos
        rows.append({
            "config_selection": json.dumps(cfg_sel),
            "config_merged":    json.dumps(merged),
            **metrics
        })

    df = pd.DataFrame(rows)
    trials_path = os.path.join(out_dir, f"random_{ticker}_trials_transformer.csv")
    df.to_csv(trials_path, index=False)
    print(f"[OK] Trials guardados → {trials_path}")

    # --------- Guardar BEST por corr y por R2 ---------
    def save_best(metric_name: str):
        best_block = search.best[metric_name]  # {"value":..., "best_configuration":..., "best_model": None}
        best_sel   = best_block["best_configuration"]
        best_val   = best_block["value"]
        best_merged = space.parameters(best_sel)

        best_path = os.path.join(out_dir, f"random_{ticker}_best_{metric_name}_transformer.json")
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

    # (Opcional) imprime resumen por consola
    print(f"\n{search.total_trials()} TRIALS")
    for (cfg_sel, evaluation) in search.trials:
        res = evaluation["oos"]
        print(cfg_sel, "→ corr=", res.metric("corr"), "R2=", res.metric("R2"))
    print("\nBEST by corr:", search.best["corr"])
    print("BEST by R2:  ", search.best["R2"])
