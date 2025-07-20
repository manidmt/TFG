from flask import Blueprint, render_template

portfolios_bp = Blueprint("portfolios", __name__)

@portfolios_bp.route("/")
def portfolio_index():
    # This will eventually render a list of all portfolios
    return render_template("es/portfolios.html") 