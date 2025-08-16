import os
from flask import Flask, redirect, url_for
from financial.momentum.web.routes.main import main_bp
from financial.momentum.web.routes.stocks import stocks_bp
from financial.momentum.web.routes.simulation import simulation_bp

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )
    app.secret_key = "change_this_to_a_secure_key"

    app.register_blueprint(main_bp, url_prefix="/<lang>")
    app.register_blueprint(stocks_bp, url_prefix="/<lang>/acciones")
    app.register_blueprint(simulation_bp, url_prefix="/<lang>/simulacion")

    # Redirect / to /es/
    @app.route("/")
    def root_redirect():
        return redirect(url_for("main.index", lang="es"))
    
    return app
