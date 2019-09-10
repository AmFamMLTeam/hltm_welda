from hltm_welda import app
from flask import render_template, url_for
from hltm_welda.api import initialize
from hltm_welda.api import corpus, lda
from hltm_welda.model.HLTM_WELDA import HLTM_WELDA
from flask.json import jsonify
import numpy as np


@app.route('/')
@app.route('/index')
def index():
    init_result = initialize()
    return render_template(
        'base.html'
    )
