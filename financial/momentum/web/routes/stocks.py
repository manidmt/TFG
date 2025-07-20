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
    details = load_model_details(model_id)
    return render_template(f"{lang}/model_detail.html", lang=lang, ticker=ticker, model_id=model_id, details=details)
