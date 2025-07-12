from flask import Flask
from routes import setup_routes
import os

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['CHECKPOINT_FOLDER'] = os.path.join(os.getcwd(), 'model', 'checkpoints')
    app.config['DATA_FOLDER'] = os.path.join(os.getcwd(), 'data', 'processed')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

    setup_routes(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
