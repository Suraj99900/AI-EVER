#!/usr/bin/env python3
# run.py

# Import all route blueprints
from routes.extract import bp_extract
from routes.infer import bp_infer
from routes.train import bp_train
from routes.home import bp_home 

import os
from flask import Flask

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # Register your routes blueprint
    app.register_blueprint(bp_extract)
    app.register_blueprint(bp_infer)
    app.register_blueprint(bp_train)
    app.register_blueprint(bp_home) 

    return app


if __name__ == "__main__":
    app = create_app()
    # you can set FLASK_ENV=production or development in env vars
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
