#!/usr/bin/env python3
import os
from flask import Flask
from models.log_manager import LogManager

# Initialize LogManager and create named loggers + handlers
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
logmgr = LogManager(base_log_dir=LOG_DIR)

# create named loggers (these create per-name files: app.log, extract.log, model_trainer.log)
logmgr.setup_logger("app", max_bytes=5 * 1024 * 1024, backup_count=5)
logmgr.setup_logger("extract", max_bytes=10 * 1024 * 1024, backup_count=3)
logmgr.setup_logger("model_trainer", max_bytes=20 * 1024 * 1024, backup_count=5)

from routes.extract import bp_extract
from routes.infer import bp_infer
from routes.train import bp_train
from routes.home import bp_home

def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # make the LogManager available via app for convenience
    app.config["log_manager"] = logmgr

    # register blueprints
    app.register_blueprint(bp_extract)
    app.register_blueprint(bp_infer)
    app.register_blueprint(bp_train)
    app.register_blueprint(bp_home)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
