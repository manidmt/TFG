import os
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras import backend as K
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import r2_score
from financial.lab.experiment import Experiment

def find_dotenv(start_path='.'):
    """
    Find the .env file by searching upwards from the start_path.
    """
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:  # Avoid the root directory
        if (current_path / '.env').is_file():
            return str(current_path / '.env')
        current_path = current_path.parent
    return None

def metrics(experiment, predictions, target, model_path, global_model=True):
    """
    Compute and store metrics for the given experiment.
    """
    all_metrics_to_save: Dict[str, Any] = {}
    
    if target.size == predictions.size:
        metrics_multiple = Experiment(experiment.name, predictions, target)
        r2 = r2_score(target, predictions)
        
        print("GLOBAL:")
        print(f"n={metrics_multiple.samples()} MSE={metrics_multiple.MSE():.4f} RMSE={metrics_multiple.RMSE():.4f} MAE={metrics_multiple.MAE():.4f} MAPE={metrics_multiple.MAPE():.4f} R² = {r2:.4f}")
        
        global_metrics = {
            "n": metrics_multiple.samples(),
            "MSE": metrics_multiple.MSE(),
            "RMSE": metrics_multiple.RMSE(),
            "MAE": metrics_multiple.MAE(),
            "MAPE": metrics_multiple.MAPE(),
            "R2": r2
        }
        all_metrics_to_save["global"] = global_metrics
    
    if global_model:
        print("Test and train metrics are being saved.")
        if experiment.train.samples() > 0:
            print("TRAIN: ")
            train_results_obj = experiment.train
            train_metrics = {
                "n": train_results_obj.samples(),
                "MSE": train_results_obj.MSE(),
                "RMSE": train_results_obj.RMSE(),
                "MAE": train_results_obj.MAE(),
                "MAPE": train_results_obj.MAPE()
            }
            print(f"n={train_metrics['n']} MSE={train_metrics['MSE']:.4f} RMSE={train_metrics['RMSE']:.4f} MAE={train_metrics['MAE']:.4f} MAPE={train_metrics['MAPE']:.4f}")
            all_metrics_to_save["train"] = train_metrics
        
        if experiment.test.samples() > 0:
            print("TEST: ")
            test_results_obj = experiment.test
            test_metrics = {
                "n": test_results_obj.samples(),
                "MSE": test_results_obj.MSE(),
                "RMSE": test_results_obj.RMSE(),
                "MAE": test_results_obj.MAE(),
                "MAPE": test_results_obj.MAPE()
            }
            print(f"n={test_metrics['n']} MSE={test_metrics['MSE']:.4f} RMSE={test_metrics['RMSE']:.4f} MAE={test_metrics['MAE']:.4f} MAPE={test_metrics['MAPE']:.4f}")
            all_metrics_to_save["test"] = test_metrics
        
    os.makedirs(model_path, exist_ok=True)
    
    # Metrics name
    file_name = f"{experiment.name}_metrics.json"
    file_path = os.path.join(model_path, file_name)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(all_metrics_to_save, f, indent=4)
        print(f"\nMetrics saved successfully in: {file_path}")
    except Exception as e:
        print(f"\nError while saving matrics in {file_path}: {e}")   



import requests

def send_telegram_message(message):
    """
    Send a message via Telegram bot.
    """
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
    except Exception as e:
        print(f"Telegram message error: {e}")
    


def reset_gpu():
    K.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()


import pickle

def store_results(ticker: str, model_name: str, predictions: pd.Series, predictions_rel: pd.Series, cache_path: str, model_path: str):
    """
    Store model predictions in both pickle and CSV formats.
    """
    for prefix in ["keras_", "scikit-learn_"]:
        rel_model_name = model_name.removeprefix(prefix)
        
    base_name = f"model-momentum-{rel_model_name}@pred" 

    with open(os.path.join(cache_path, base_name), "wb") as f:
        pickle.dump(predictions_rel, f)
    print("Predictions saved pickle")

    predictions.to_csv(os.path.join(model_path, model_name + "_preds.csv"))
    print("Predictions saved csv")


if __name__ == '__main__':

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
        model_path = os.getenv("MODEL")
        print(f"Model path loaded from: {dotenv_path}")
        print(f"MODEL: {model_path}")
    else:
        print("No .env file found.")