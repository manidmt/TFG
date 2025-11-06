'''
Creating random search trials for hyperparameter tuning of financial momentum models.

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
from financial.momentum.tuning.hyperSpaces import build_transformer_space, build_cnn_space, build_lstm_space


if __name__ == "__main__":
    load_dotenv()

    # --------- Parameters ----------
    ticker     = "^GSPC"
    start_date = "1990-01-01"
    end_date   = "2025-06-30"
    lookahead  = 20
    horizon    = 90
    trials     = 15
    architecture = "cnn"  # "transformer", "cnn", "lstm"

    out_dir = os.path.join(os.environ["MODEL"], "tuning")
    os.makedirs(out_dir, exist_ok=True)

    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )

    # --------- Space + Evaluator ----------
    if architecture == "transformer":
        space = build_transformer_space()
    elif architecture == "cnn":
        space = build_cnn_space()
    elif architecture == "lstm":
        space = build_lstm_space()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    evaluator = OOSMomentumEvaluator(
        ds=ds, ticker=ticker,
        start=start_date, end=end_date,
        lookahead=lookahead, horizon=horizon,
        quick_oos_metrics_fn=quick_oos_metrics,
        default_architecture=architecture
    )

    factory = KerasAdvancedModelFactory()
    search = RandomSearch(space, ds, factory, evaluator, trials=trials)
    # MMetrics used by RandomSearch to decide "best"
    search.metrics = ["corr", "R2", "-MAE", "-RMSE", "hit_rate"]
    search.best = {
        m: {"value": float("-inf"), "best_configuration": None, "best_model": None}
        for m in search.metrics
    }
    # --------- Execute ----------
    search.run()

    # --------- Save ALL trials ----------
    rows = []
    for (cfg_sel, evaluation) in search.trials:
        metrics = evaluation["oos"].m
        merged  = space.parameters(cfg_sel)  # full merged configuration
        rows.append({
            "config_selection": json.dumps(cfg_sel),
            "config_merged":    json.dumps(merged),
            **metrics
        })

    df = pd.DataFrame(rows)
    trials_path = os.path.join(out_dir, f"random_{ticker}_trials_{architecture}.csv")
    df.to_csv(trials_path, index=False)
    print(f"[OK] Trials saved → {trials_path}")

    # --------- Save BEST by corr and by R2 ---------
    def save_best(metric_name: str):
        best_block = search.best[metric_name]  # {"value":..., "best_configuration":..., "best_model": None}
        best_sel   = best_block["best_configuration"]
        best_val   = best_block["value"]
        best_merged = space.parameters(best_sel)

        best_path = os.path.join(out_dir, f"random_{ticker}_best_{metric_name}_{architecture}.json")
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
    print(f"\n{search.total_trials()} TRIALS")
    for (cfg_sel, evaluation) in search.trials:
        res = evaluation["oos"]
        print(cfg_sel, "→ corr=", res.metric("corr"), "R2=", res.metric("R2"))
    print("\nBEST by corr:", search.best["corr"])
    print("BEST by R2:  ", search.best["R2"])
