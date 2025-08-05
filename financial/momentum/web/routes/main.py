from flask import Blueprint, render_template    
from financial.momentum.web.utils.loader import get_recent_models_separated

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index(lang):
    keras_models, sklearn_models = get_recent_models_separated()
    #print(f"sklearn_models: {sklearn_models}")
    #print(f"keras_models: {keras_models}")
    return render_template(f"{lang}/index.html", 
                           lang=lang, 
                           keras_models=keras_models, 
                           sklearn_models=sklearn_models)
