# routes/home.py

from flask import Blueprint, render_template

bp_home = Blueprint("home", __name__)

@bp_home.route("/")
def index():
    return render_template("index.html")

