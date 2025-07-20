import os
import re
from datetime import datetime

import json
import pandas as pd
import xml.etree.ElementTree as ET

def parse_model_filename(name):
    '''
    Extract model metadata from filenames like:
    cnn_GOOG_2025_single
    lstm_^GSPC_2025_single
    '''
    pattern = r'^([a-zA-Z0-9]+)_([\^A-Z0-9]+)_(\d{4})_(single|multiple)(?:_(.*?))?_(metrics|preds|hyperparameters|metadata)\.(json|csv|xml|keras|h5|pickle)$'
    match = re.match(pattern, name)
    if not match:
        return None


    architecture, ticker, year, type_, extra, suffix, extension = match.groups()
    return {
        'architecture': architecture,
        'ticker': ticker,
        'year': year,
        'type': type_,
        'extra': extra or "",
        'suffix': suffix,
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

