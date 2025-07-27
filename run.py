#!/usr/bin/env python3
# run.py

import os
from flask import Flask
from routes import bp  # your Blueprint instance

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # Basic configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret"),
        UPLOAD_FOLDER=os.path.join(os.getcwd(), "uploads"),
        CHECKPOINT_FOLDER=os.path.join(os.getcwd(), "model", "checkpoints"),
        DATA_FOLDER=os.path.join(os.getcwd(), "data", "processed"),
    )

    # Ensure directories exist
    for key in ("UPLOAD_FOLDER", "CHECKPOINT_FOLDER", "DATA_FOLDER"):
        os.makedirs(app.config[key], exist_ok=True)

    # Register your routes blueprint
    app.register_blueprint(bp)

    return app


if __name__ == "__main__":
    app = create_app()
    # you can set FLASK_ENV=production or development in env vars
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
