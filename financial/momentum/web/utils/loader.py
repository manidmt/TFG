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
        r'([\^A-Z0-9]+)'                         # main ticker
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

    return {
        'framework': framework,
        'architecture': architecture,
        'ticker': main_ticker,
        'extra_tickers': extra_tickers.split("_") if extra_tickers else [],
        'year': year,
        'type': type_ or "unknown",  # o "single" si quieres asumir por defecto
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
    directory = directory or os.environ.get("MODEL")
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

def generate_plot(ticker, preds_df, lang="es"):
    # 1. Carga serie original
    ds = fd.CachedDataStore(
        path=os.environ["DATA"],
        cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy())
    )
    data = ds.get_data(ticker)

    # 2. Recorta la serie real al rango del CSV
    # real = data.loc[preds_df["Date"]].copy()
    # real = real.to_frame(name="Real")

    # # 3. Une predicción y valor real
    # combined = pd.merge(preds_df, real[["Real"]], left_on="Date", right_index=True)
     # Asegura DataFrame con índice datetime limpio
    # if isinstance(data, pd.Series):
    #     data = data.to_frame(name="Real")
    # else:
    #     # Si hay varias columnas, intenta coger 'Close' o la 1ª numérica
    #     if "Real" not in data.columns:
    #         if "Close" in data.columns:
    #             data = data[["Close"]].rename(columns={"Close": "Real"})
    #         else:
    #             num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    #             if not num_cols:
    #                 raise ValueError("Data for ticker has no numeric columns to plot.")
    #             data = data[[num_cols[0]]].rename(columns={num_cols[0]: "Real"})

    # data = data.copy()

    # data.index = pd.to_datetime(data.index, errors="coerce").tz_localize(None).normalize()
    # data = data.dropna(subset=["Real"]).sort_index()

    # # 2) Normaliza preds_df (por si acaso)
    # preds = preds_df.copy()
    # preds["Date"] = pd.to_datetime(preds["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    # preds = preds.dropna(subset=["Date"]).sort_values("Date")

    # # Detecta columna de predicción
    # pred_cols = [c for c in preds.columns if c != "Date"]
    # if not pred_cols:
    #     raise ValueError("preds_df must contain a prediction column besides 'Date'.")
    # # Preferimos 'Prediction' si existe
    # if "Prediction" in pred_cols:
    #     pred_col = "Prediction"
    # else:
    #     # coge la primera numérica
    #     numeric_candidates = [c for c in pred_cols if pd.api.types.is_numeric_dtype(preds[c])]
    #     pred_col = numeric_candidates[0] if numeric_candidates else pred_cols[0]
    #     if pred_col != "Prediction":
    #         preds = preds.rename(columns={pred_col: "Prediction"})
    #         pred_col = "Prediction"

    # # 3) Alinea fechas (domingo/fiesta -> último día de mercado anterior)
    # mkt_reset = data.reset_index().rename(columns={"index": "Date"})  # Date + Real
    # combined = pd.merge_asof(
    #     preds.sort_values("Date"),
    #     mkt_reset.sort_values("Date"),
    #     on="Date",
    #     direction="backward",                 # usa el cierre anterior
    #     tolerance=pd.Timedelta("10D"),        # hasta 10 días hacia atrás
    # ).dropna(subset=["Real"])

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
    mkt_reset = data.reset_index().rename(columns={"index": "Date"})
    preds = preds_df.sort_values("Date").copy()
    combined = pd.merge_asof(
        preds, mkt_reset, on="Date", direction="backward", tolerance=pd.Timedelta("10D")
    ).dropna(subset=["Real"])



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

                        if model_info["framework"] == "keras":
                            keras_models.append(model_info)
                        elif model_info["framework"] == "scikit-learn":
                            sklearn_models.append(model_info)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    keras_models = sorted(keras_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]
    sklearn_models = sorted(sklearn_models, key=lambda x: x["timestamp"], reverse=True)[:max_models]

    return keras_models, sklearn_models



