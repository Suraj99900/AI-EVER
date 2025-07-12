from flask import render_template, request, jsonify
from training import start_training
from inference import run_inference

def setup_routes(app):

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/train", methods=["GET", "POST"])
    def train():
        if request.method == "POST":
            # e.g. collect form fields for hyperparams
            config = {
                "epochs": int(request.form.get("epochs", 3)),
                "batch_size": int(request.form.get("batch_size", 1)),
                # add more as needed...
            }
            status = start_training(config)
            return jsonify(status=status)
        return render_template("train.html")

    @app.route("/inference", methods=["GET", "POST"])
    def inference():
        if request.method == "POST":
            payload = request.get_json(force=True)
            prompt = payload.get("prompt", "")
            params = {
                "max_new_tokens": int(payload.get("max_new_tokens", 512)),
                "temperature": float(payload.get("temperature", 0.7)),
                "top_p": float(payload.get("top_p", 0.95))
            }
            output = run_inference(prompt, params)
            return jsonify(output=output)
        return render_template("inference.html")
