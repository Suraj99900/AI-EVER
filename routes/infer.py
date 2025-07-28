import os
from flask import Blueprint, request, jsonify, render_template, current_app
from models.ModelInference import ModelInference
from sql.AIEverLog import AIEverLog

bp_infer = Blueprint("infer", __name__)

# Model instance should ideally be loaded once (singleton-like behavior)
model = None
log = AIEverLog()


def load_model():
    global model
    try:
        if not model:
            model = ModelInference()
            pass
    except Exception as e:
        log.log_error("load_model", str(e))


@bp_infer.route("/inference", methods=["GET", "POST"])
def infer():
    load_model()

    if request.method == "POST":
        input_text = request.form.get("prompt", "").strip()
        max_new_tokens = int(request.form.get("max_new_tokens", 128))
        temperature = float(request.form.get("temperature", 0.7))

        if not input_text:
            return jsonify(status="error", message="Prompt is required"), 400

        try:
            result = model.infer(
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return jsonify(status="success", result=result)
        except Exception as e:
            log.log_error("inference", str(e))
            return jsonify(status="error", message=str(e)), 500

    return render_template("inference.html")
