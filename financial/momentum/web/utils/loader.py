import os
import re
from datetime import datetime

import json
import pandas as pd
import xml.etree.ElementTree as ET

import re

def parse_model_filename(name):
    '''
    Parses filenames like:
    keras_cnn_AAPL_2025_single_metrics.json
    keras_cnn_AAPL_^GSPC_2025_multiple_metrics.json
    scikit-learn_randomforest_AAPL_2025_single_metrics.json
    '''
    pattern = (
        r'^(keras|scikit-learn)_'            # framework
        r'([a-zA-Z0-9]+)_'                   # architecture
        r'([\^A-Z0-9]+)'                     # main ticker
        r'(?:_([\^A-Z0-9_]+))?'              # optional extra tickers
        r'_(\d{4})_'                         # year
        r'(single|multiple)_'                # type
        r'(metrics|preds|hyperparameters|metadata)\.'  # filetype
        r'(json|csv|xml|keras|h5|pickle)$'   # extension
    )

    match = re.match(pattern, name)
    if not match:
        return None

    framework, architecture, main_ticker, extra_tickers, year, type_, filetype, extension = match.groups()

    return {
        'framework': framework,
        'architecture': architecture,
        'ticker': main_ticker,
        'extra_tickers': extra_tickers.split("_") if extra_tickers else [],
        'year': year,
        'type': type_,
        'filetype': filetype,
        'extension': extension,
        'filename': name
    }


def list_recent_models(directory=None, max_results=10):
    '''
    List most recent models based on their metric file timestamps
    '''
    directory = directory or os.environ.get("MODEL")
    models = []

    for file in os.listdir(directory):
        if file.endswith("_metrics.json"):
            data = parse_model_filename(file)
            if data:
                path = os.path.join(directory, file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                data["timestamp"] = mod_time
                models.append(data)

    models.sort(key=lambda x: x["timestamp"], reverse=True)
    return models[:max_results]

def list_tickers(directory=None):
    '''
    Return unique tickers that have at least one model stored
    '''
    directory = directory or os.environ.get("MODEL")
    print(f"Looking in directory:{directory}")
    tickers = set()

    for filename in os.listdir(directory):
        if "_metrics.json" in filename:
            data = parse_model_filename(filename)
            if data:
                tickers.add(data["ticker"])

    return sorted(tickers)

def list_models_for_ticker(ticker, directory=None):
    '''
    Return all model files (with metadata) for a given ticker,
    including the base filename without suffix (for linking).
    '''
    directory = directory or os.environ.get("MODEL")
    models = []

    for filename in os.listdir(directory):
        if "_metrics.json" in filename:
            data = parse_model_filename(filename)
            if data and data["ticker"] == ticker:
                data["base_name"] = filename.replace("_metrics.json", "")
                models.append(data)

    return sorted(models, key=lambda x: (x["year"], x["architecture"]))

def load_model_details(model_id, directory=None):
    '''
    Load metrics, predictions, hyperparameters, and metadata for a given model
    '''
    directory = directory or os.environ.get("MODEL")
    base = os.path.join(directory, model_id)

    # Load metrics
    try:
        with open(base + "_metrics.json", "r") as f:
            metrics = json.load(f)
    except:
        metrics = {}

    # Load predictions
    try:
        preds = pd.read_csv(base + "_preds.csv")
        predictions = preds.to_dict(orient="records")
    except:
        predictions = []

    # Load hyperparameters
    try:
        with open(base + ".hyperparameters.json", "r") as f:
            hyperparams = json.load(f)
    except:
        hyperparams = {}

    # Load metadata (XML)
    try:
        tree = ET.parse(base + ".xml")
        root = tree.getroot()
        metadata = {elem.tag: elem.text for elem in root.iter()}
    except:
        metadata = {}

    return {
        "metrics": metrics,
        "predictions": predictions,
        "hyperparameters": hyperparams,
        "metadata": metadata
    }

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from financial.io.cache import NoUpdateStrategy
from financial.io.file.cache import FileCache
import financial.data as fd

def generate_plot(ticker, preds_df, lang="es"):
    # 1. Carga serie original
    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )
    data = ds.get_data(ticker)

    # 2. Recorta la serie real al rango del CSV
    real = data.loc[preds_df["Date"]].copy()
    real = real.to_frame(name="Real")

    # 3. Une predicción y valor real
    combined = pd.merge(preds_df, real[["Real"]], left_on="Date", right_index=True)

    # 4. Gráfico interactivo

    if lang == "en":
        label_6m = "6m"
        label_1y = "1y"
        label_3y = "3y"
        label_all = "All"
        title = "Real vs Prediction"
        xaxis_title = "Date"
        yaxis_title = "Value"
    else:
        label_6m = "6m"
        label_1y = "1a"
        label_3y = "3a"
        label_all = "Todo"
        title = "Gráfico Real vs Predicción"
        xaxis_title = "Fecha"
        yaxis_title = "Valor"

    # Gráfico
    fig = go.Figure()
    legend_real = "Real" if lang == "en" else "Real"
    legend_pred = "Prediction" if lang == "en" else "Predicción"

    fig.add_trace(go.Scatter(x=combined["Date"], y=combined["Real"], name=legend_real, line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=combined["Date"], y=combined.iloc[:, 1], name=legend_pred, line=dict(color="red")))

    fig.update_layout(
        title=title,
        height=600,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label=label_6m, step="month", stepmode="backward"),
                    dict(count=1, label=label_1y, step="year", stepmode="backward"),
                    dict(count=3, label=label_3y, step="year", stepmode="backward"),
                    dict(step="all", label=label_all)
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return pio.to_html(fig, full_html=False)



def get_recent_models_separated(directories=None, max_models=15):
    if directories is None:
        directories = [
            "/home/manidmt/TFG/OTRI/models/keras/",
            "/home/manidmt/TFG/OTRI/models/scikit-learn/"
        ]

    keras_models = []
    sklearn_models = []

    for directory in directories:
        for filename in os.listdir(directory):
            if re.search(r'_metrics\.json$', filename):
                full_path = os.path.join(directory, filename)
                try:
                    timestamp = os.path.getmtime(full_path)
                    model_info = parse_model_filename(filename)
                    if model_info:
                        model_info["timestamp"] = timestamp
                        model_info["filename"] = filename
                        model_info["model_id"] = filename.replace("_metrics.json", "")
                        
                        if "keras" in directory:
                            keras_models.append(model_info)
                        elif "scikit-learn" in directory:
                            sklearn_models.append(model_info)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    keras_models = sorted(keras_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]
    sklearn_models = sorted(sklearn_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]

    return keras_models, sklearn_models



