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


CHECKPOINTS = [
    {"id": "2025-07-01", "title": "30 Days"},
    {"id": "2025-05-23", "title": "Optimizing Training Script"},
    {"id": "2025-03-29", "title": "Video Player UI"},
]


@bp_infer.route("/inference", methods=["GET", "POST"])
def infer():
    load_model()

    if request.method == "POST":
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify(status="error", message="Prompt is required"), 400

        # Making prompt properly in the template 
        prompt = '### Instruction:\n' + prompt + '\n### Response:'
        print(f"Received prompt: {prompt}")
        try:
            result = model.generate_response(
                prompt=prompt,
                max_new_tokens=int(data.get("max_new_tokens", 700)),
                temperature=float(data.get("temperature", 0.2)),
                top_p=float(data.get("top_p", 0.95)),
                repetition_penalty=float(data.get("repetition_penalty", 1.2)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                num_beams=int(data.get("num_beams", 1)),
            )
            return jsonify(status="success", result=result)
        except Exception as e:
            log.log_error("inference", str(e))
            return jsonify(status="error", message=str(e)), 500

    # on GET, render the chat page and pass your saved checkpoints
    return render_template("inference.html", checkpoints=CHECKPOINTS)
