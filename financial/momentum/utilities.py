import os
from dotenv import load_dotenv
from pathlib import Path

def find_dotenv(start_path='.'):
    """Busca el archivo .env ascendiendo desde la ruta de inicio."""
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:  # Evita el directorio raíz
        if (current_path / '.env').is_file():
            return str(current_path / '.env')
        current_path = current_path.parent
    return None

# Prueba de carga del archivo .env

import pandas as pd
import numpy as np
import tensorflow as tf # Necesario para tf.convert_to_tensor si decides usarlo, aunque la propuesta actual usa solo NumPy.

# Importa tus otras dependencias (labevaluation, fd, etc.)
# from ... import ModelBuildingTask # Asumo que ModelBuildingTask es una clase base
# import ... # Asegúrate de importar fd, ModelFactory, etc.

# --- FUNCIÓN AUXILIAR: _create_sequences_from_df (Ponla al principio del archivo o en un archivo de utilidades) ---
def _create_sequences_from_df(df_features: pd.DataFrame, df_target: pd.DataFrame, horizon: int):
    """
    Convierte un DataFrame 2D de features (con N columnas de 'change_i')
    y un DataFrame/Series de target en arrays NumPy 3D (para features) y 2D (para target).

    Args:
        df_features (pd.DataFrame): DataFrame con todas las features (e.g., 90 columnas para 90 cambios).
        df_target (pd.DataFrame or pd.Series): DataFrame/Series con el target.
        horizon (int): El número de timesteps (e.g., 90).

    Returns:
        tuple: (X_reshaped_np, y_np)
    """
    X_np = df_features.values # Convertir a NumPy array
    y_np = df_target.values   # Convertir a NumPy array

    n_samples, n_features_total = X_np.shape
    assert n_features_total % horizon == 0, (
        f"Incompatible shape: expected features to be divisible by horizon={horizon}, "
        f"but got total={n_features_total}"
    )
    n_features_per_timestep = n_features_total // horizon # Debería ser 1 en tu caso

    # Reshape the features to (n_samples, timesteps, n_features_per_timestep)
    X_reshaped_np = X_np.reshape((n_samples, horizon, n_features_per_timestep))

    # Asegurarse de que y_np tenga la forma (n_samples, 1) para Keras
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    return X_reshaped_np, y_np

import json
from typing import Dict, Any
from sklearn.metrics import r2_score
from financial.lab.experiment import Experiment

def metrics(experiment, predictions, target):
    metrics_multiple = Experiment(experiment.name, predictions, target)
    r2 = r2_score(target, predictions)
    
    all_metrics_to_save: Dict[str, Any] = {}
    
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
    
    # Construir el nombre del archivo
    file_name = f"{experiment.name}_metrics.json"
    file_path = os.path.join(model_path, file_name)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(all_metrics_to_save, f, indent=4)
        print(f"\nMétricas guardadas exitosamente en: {file_path}")
    except Exception as e:
        print(f"\nError al guardar las métricas en el archivo {file_path}: {e}")   





if __name__ == '__main__':

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
        model_path = os.getenv("MODEL")
        print(f"Ruta del modelo cargada desde: {dotenv_path}")
        print(f"MODEL: {model_path}")
    else:
        print("No se encontró el archivo .env.")