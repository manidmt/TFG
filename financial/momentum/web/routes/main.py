from flask import Blueprint, render_template, redirect, url_for
from financial.momentum.web.utils.loader import list_recent_models

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index(lang):
    models = list_recent_models()
    return render_template(f"{lang}/index.html", models=models, lang=lang)
