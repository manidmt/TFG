from flask import Blueprint, render_template
import os
from financial.momentum.web.utils.loader import list_tickers
stocks_bp = Blueprint("stocks", __name__)

@stocks_bp.route("/")
def stock_list(lang):
    tickers = list_tickers()
    return render_template(f"{lang}/stocks.html", tickers=tickers, lang=lang)

@stocks_bp.route("/<ticker>")
def models_by_ticker(lang, ticker):
    from financial.momentum.web.utils.loader import list_models_for_ticker
    models = list_models_for_ticker(ticker)
    return render_template(f"{lang}/stocks_ticker.html", ticker=ticker, models=models, lang=lang)

@stocks_bp.route("/<ticker>/<model_id>")
def model_detail(lang, ticker, model_id):
    from financial.momentum.web.utils.loader import load_model_details
    from financial.momentum.web.utils.loader import generate_plot
    import pandas as pd

    def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
        # Si ya existe 'Date', perfecto
        if "Date" in df.columns:
            pass
        else:
            # Candidatas típicas cuando se guarda el índice sin nombre
            candidates = [c for c in df.columns
                          if (str(c).strip() == "" or
                              str(c).lower() in ("date", "fecha") or
                              str(c).startswith("Unnamed"))]
            if candidates:
                df = df.rename(columns={candidates[0]: "Date"})
            else:
                # A veces la fecha viene en el índice
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df = df.reset_index().rename(columns={"index": "Date"})
                else:
                    # Intento de parsear el índice a fecha
                    try:
                        pd.to_datetime(df.index, errors="raise")
                        df = df.reset_index().rename(columns={"index": "Date"})
                    except Exception:
                        # Último recurso: primera columna a 'Date'
                        first_col = df.columns[0]
                        df = df.rename(columns={first_col: "Date"})
        # Parseo de fechas robusto y limpieza
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
        df = df.dropna(subset=["Date"]).sort_values("Date")
        return df

    details = load_model_details(model_id)
    if details["predictions"]:
        preds_df = pd.DataFrame(details["predictions"])
        preds_df = ensure_date_column(preds_df)
        #preds_df = preds_df.sort_values("Date")
        plot_html = generate_plot(ticker, preds_df, lang=lang)
    else:
        plot_html = None
    return render_template(f"{lang}/model_detail.html", lang=lang, ticker=ticker, model_id=model_id, details=details, plot_html=plot_html)
