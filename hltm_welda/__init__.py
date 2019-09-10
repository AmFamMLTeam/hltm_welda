from flask import Flask
from flask_cors import CORS, cross_origin
import dash_bootstrap_components as dbc


app = Flask(
    import_name=__name__,
)
CORS(app)

from hltm_welda import routes, api, model 
