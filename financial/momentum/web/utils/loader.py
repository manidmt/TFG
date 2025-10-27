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
    scikit-learn_randomforest_AAPL_2025_metrics.json
    '''
    pattern = (
        r'^(keras|scikit-learn)_'                # framework
        r'([a-zA-Z0-9]+)_'                       # architecture
        r'([\^A-Z0-9\-]+)'                       # main ticker
        r'(?:_([\^A-Z0-9_]+))?'                  # optional extra tickers
        r'_(\d{4})'                              # year
        r'(?:_(single|multiple))?'               # optional type
        r'_(metrics|preds|hyperparameters|metadata)\.'  # filetype
        r'(json|csv|xml|keras|h5|pickle)$'       # extension
    )

    match = re.match(pattern, name)
    if not match:
        return None

    framework, architecture, main_ticker, extra_tickers, year, type_, filetype, extension = match.groups()

    if extra_tickers:
        tickers = extra_tickers.split("_")
        extras = " + ".join(display_name_for(t) for t in tickers)
    else:
        extras = ""

    return {
        'framework': framework,
        'architecture': architecture,
        'ticker': main_ticker,
        'extra_tickers': extras,
        'year': year,
        'type': type_ or "unknown",
        'filetype': filetype,
        'extension': extension,
        'filename': name
    }

def _model_dirs():
    return [
        "/home/manidmt/TFG/OTRI/models/keras/",
        "/home/manidmt/TFG/OTRI/models/scikit-learn/",
    ]


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
    tickers = set()
    for directory in _model_dirs():
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
    models = []
    for directory in _model_dirs():
        for filename in os.listdir(directory):
            if "_metrics.json" in filename:
                data = parse_model_filename(filename)
                if data and data["ticker"] == ticker:
                    data["base_name"] = filename.replace("_metrics.json", "")
                    models.append(data)

    return sorted(models, key=lambda x: (x["year"], x["architecture"]))

def list_extra_tickers_for_arch(arch: str):
    """
    """
    if not arch:
        return []

    directory = "/home/manidmt/TFG/OTRI/models/keras/"

    pattern = re.compile(
        rf'^keras_{re.escape(arch)}_([\^A-Z0-9]+)_([\^A-Z0-9]+)_(\d{{4}})_multiple_'
    )

    extras = set()
    try:
        for filename in os.listdir(directory):
            # Rápido filtro para evitar parsear todo
            if not (filename.startswith(f"keras_{arch}_") and "_multiple_" in filename):
                continue

            m = pattern.match(filename)
            if m:
                # group(2) es el EXTRA
                extra = m.group(2)
                extras.add(extra)
    except FileNotFoundError:
        pass

    return sorted(extras)

def load_model_details(model_id, directory=None):
    '''
    Load metrics, predictions, hyperparameters, and metadata for a given model
    '''
    if "svr" in model_id or "randomforest" in model_id or "clenow" in model_id:
        directory = _model_dirs()[1]
    else:
        directory = _model_dirs()[0]

    base = os.path.join(directory, model_id)

    # Load metrics
    try:
        with open(base + "_metrics.json", "r") as f:
            metrics = json.load(f)
    except:
        metrics = {}

    try:
        preds = pd.read_csv(base + "_preds.csv")
        if "Date" not in preds.columns:
            preds = preds.rename(columns={preds.columns[0]: "Date"})
        preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce")
        preds = preds.dropna(subset=["Date"]).sort_values("Date")
        predictions = preds.to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar predicciones: {e}")
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

def align_to_market(preds: pd.DataFrame, data: pd.DataFrame, mode="backward", tol="10D"):
    preds = preds.sort_values("Date").copy()
    mkt = data.copy()
    mkt.index = pd.to_datetime(mkt.index).tz_localize(None).normalize()
    mkt = mkt.sort_index()
    mkt_reset = mkt.reset_index().rename(columns={"index": "Date"})
    out = pd.merge_asof(
        preds, mkt_reset, on="Date",
        direction=mode, tolerance=pd.Timedelta(tol)
    )
    return out.dropna(subset=["Real"])

def generate_plot(ticker, preds_df, lang="es"):
    # 1. Carga serie original
    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )
    data = ds.get_data(ticker)

    if isinstance(data, pd.Series):
        data = data.to_frame(name="Real")
    elif "Real" not in data.columns:
        if "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": "Real"})
        else:
            num = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            data = data[[num[0]]].rename(columns={num[0]: "Real"})
    data = data.sort_index()

    # Alinear fechas (merge_asof backward)
    # mkt_reset = data.reset_index().rename(columns={"index": "Date"})
    # preds = preds_df.sort_values("Date").copy()
    # combined = pd.merge_asof(
    #     preds, mkt_reset, on="Date", direction="backward", tolerance=pd.Timedelta("10D")
    # ).dropna(subset=["Real"])

    preds = preds_df.copy()
    preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce")
    preds = preds.dropna(subset=["Date"]).sort_values("Date")

    combined = align_to_market(preds, data, mode="backward", tol="10D")

    # 4. Gráfico interactivo

    if lang == "en":
        label_6m = "6m"
        label_1y = "1y"
        label_3y = "3y"
        label_all = "All"
        title = "Real vs Prediction"
        xaxis_title = "Date"
        yaxis_title = "Value"
        legend_pred = "Prediction"
    else:
        label_6m = "6m"
        label_1y = "1a"
        label_3y = "3a"
        label_all = "Todo"
        title = "Gráfico Real vs Predicción"
        xaxis_title = "Fecha"
        yaxis_title = "Valor"
        legend_pred = "Predicción"


    # Gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=combined["Date"], y=combined["Real"], name="Real", line=dict(color="blue")))
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



def get_recent_models_separated(max_models=15):

    keras_models = []
    sklearn_models = []

    for directory in _model_dirs():
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

                        if model_info["framework"] == "keras":
                            keras_models.append(model_info)
                        elif model_info["framework"] == "scikit-learn":
                            sklearn_models.append(model_info)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    keras_models = sorted(keras_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]
    sklearn_models = sorted(sklearn_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]

    return keras_models, sklearn_models


from functools import lru_cache
import xml.etree.ElementTree as ET

@lru_cache()
def _ticker_name_map(xml_path=None):
    xml_path = xml_path or os.environ.get("DATA_XML", "/home/manidmt/TFG/OTRI/data/data.xml")
    mapping = {}
    try:
        root = ET.parse(xml_path).getroot()
        for var in root.findall("variable"):
            t = (var.findtext("ticker") or "").strip()
            d = (var.findtext("description") or "").strip()
            if t and d:
                mapping[t] = d
    except Exception:
        pass

    return mapping

def display_name_for(ticker: str) -> str:
    print(ticker)
    return _ticker_name_map().get(ticker, ticker)



