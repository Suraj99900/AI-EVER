import os
from flask import Blueprint, request, jsonify, render_template, current_app
from models.ModelInference import ModelInference
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog

bp_infer = Blueprint("infer", __name__)

# Model instance should ideally be loaded once (singleton-like behavior)
model = None



def load_model():
    global model
    try:
        if not model:
            model = ModelInference()
            pass
    except Exception as e:
        log.log_error("load_model", str(e))

def fetch_checkpoints():
    """Fetch available checkpoints from the model DB."""
    CHECKPOINTS = []
    try:
        # Assuming a function to fetch checkpoints from the database
        CHECKPOINTS = CheckpointTrackMaster().get_all_checkpoints()

    except Exception as e:
        log.log_error("fetch_checkpoints", str(e))
    return CHECKPOINTS


# CHECKPOINTS = [
#     {"id": "2025-07-01", "title": "30 Days"},
#     {"id": "2025-05-23", "title": "Optimizing Training Script"},
#     {"id": "2025-03-29", "title": "Video Player UI"},
# ]


@bp_infer.route("/inference", methods=["GET", "POST"])
def infer():
    load_model()
    log = AIEverLog()
    CHECKPOINTS = fetch_checkpoints()
    print(f"Available checkpoints: {CHECKPOINTS}")

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
                max_new_tokens=int(data.get("max_new_tokens", 1000)),
                temperature=float(data.get("temperature", 0.1)),
                top_p=float(data.get("top_p", 0.95)),
                repetition_penalty=float(data.get("repetition_penalty", 1.2)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                num_beams=int(data.get("num_beams", 4)),
                do_sample=False
            )
            return jsonify(status="success", result=result)
        except Exception as e:
            log.log_error("inference", str(e))
            return jsonify(status="error", message=str(e)), 500
    formatted_checkpoints = []
    seen = set()
    for cp in CHECKPOINTS:
        folder_name = os.path.basename(cp[2])
        if folder_name not in seen:
            seen.add(folder_name)
            formatted_checkpoints.append((cp[0], folder_name, cp[2]))

    # on GET, render the chat page and pass your saved checkpoints
    return render_template("inference.html", checkpoints=formatted_checkpoints)


@bp_infer.route("/rename_checkpoint/<int:cp_id>", methods=["POST"])
def rename_checkpoint(cp_id):
    data = request.get_json()
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"success": False, "error": "Invalid name"})

    try:
        cp = CheckpointTrackMaster().get_checkpoint_by_id(cp_id)  # your DB fetch
        if not cp:
            return jsonify({"success": False, "error": "Checkpoint not found"})

        old_path = cp[2]
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        os.rename(old_path, new_path)

        CheckpointTrackMaster().update_checkpoint(cp_id, new_path)  # update DB
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})