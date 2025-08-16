from flask import Blueprint, request, render_template, jsonify
from financial.momentum.web.utils.loader import list_tickers, list_extra_tickers_for_arch
from financial.momentum.web.utils.simulation import prepare_simulation_outputs

simulation_bp = Blueprint("simulation", __name__)

@simulation_bp.route("/", methods=["GET", "POST"])
def simulation(lang):
    if request.method == "POST":
        start_year   = int(request.form["start_year"])
        end_year     = int(request.form["end_year"])
        universe     = request.form.getlist("tickers")
        architecture = request.form["architecture"]
        # “Ninguno” llega como "" -> filtramos vacíos
        extra_info   = [e for e in request.form.getlist("extra_tickers") if e]
        num_assets   = int(request.form["num_assets"])
        refuge       = request.form.get("refuge") or None

        # Si la arquitectura es ML, fuerza extras vacíos
        if architecture in ("svr", "randomforest"):
            extra_info = []

        stats, plot_html = prepare_simulation_outputs(
            start_year, end_year, universe, architecture, extra_info, num_assets, refuge, lang=lang
        )
        return render_template(f"{lang}/simulation_result.html", stats=stats, graph=plot_html, lang=lang)

    tickers = list_tickers()
    return render_template(f"{lang}/simulation_form.html", tickers=tickers, lang=lang)

# API: lista de extras disponibles para una arquitectura Keras
@simulation_bp.route("/api/extras")
def api_extras(lang):
    arch = request.args.get("arch", "").strip()
    extras = list_extra_tickers_for_arch(arch) if arch else []
    return jsonify({"extras": extras})

