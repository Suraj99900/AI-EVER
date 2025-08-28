from math import log
import os
from flask import Blueprint, request, jsonify, render_template, current_app,Response, stream_with_context
from transformers import pipeline

from models.ModelInference import ModelInference
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog
import logging
from pathlib import Path

bp_infer = Blueprint("infer", __name__)

# Model instance should ideally be loaded once (singleton-like behavior)
model = None
log_dir = Path(__file__).parent.parent / "log"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "model_inference.log"

# Lodding the model...
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)


model_instance = None

def load_model():
    global model_instance
    if model_instance is None:
        try:
            logging.info("[Model Loader] Initializing ModelInference...")
            model_instance = ModelInference()
            logging.info("[Model Loader] Model loaded successfully")
        except Exception as e:
            logging.error(f"[Model Loader] Failed: {str(e)}")
            raise
    return model_instance

def fetch_checkpoints():
    """Fetch available checkpoints from the model DB."""
    CHECKPOINTS = []
    try:
        # Assuming a function to fetch checkpoints from the database
        CHECKPOINTS = CheckpointTrackMaster().get_all_checkpoints()

    except Exception as e:
        logging.error("fetch_checkpoints", str(e))
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

    if request.method == "POST":
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify(status="error", message="Prompt is required"), 400

        # Making prompt properly in the template 
        prompt = '### Instruction:\n' + prompt + '\n### Response:'
        logging.info(f"Received prompt: {prompt}")
        try:
            # result = model.generate_response(
            #     prompt=prompt,
            #     max_new_tokens=int(data.get("max_new_tokens", 1000)),
            #     temperature=float(data.get("temperature", 0.1)),
            #     top_p=float(data.get("top_p", 0.95)),
            #     repetition_penalty=float(data.get("repetition_penalty", 1.2)),
            #     no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
            #     num_beams=int(data.get("num_beams", 4)),
            #     do_sample=False
            # )
            result = model.generate_response_stream(
                prompt=prompt,
                max_new_tokens=int(data.get("max_new_tokens", 1000)),
                temperature=float(data.get("temperature", 0.2)),
                top_p=float(data.get("top_p", 0.9)),
                repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                do_sample=False,
                num_beams=int(data.get("num_beams", 4)),
            )
            return jsonify(status="success", result=result)
        except Exception as e:
            logging.error("inference", str(e))
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

@bp_infer.route("/stream_inference", methods=["POST"])
def stream_inference():
    load_model()
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify(status="error", message="Prompt is required"), 400

    prompt = '### Instruction:\n' + prompt + '\n### Response:'
    logging.info(f"Received streaming prompt: {prompt}")

    try:
        # Generator that yields tokens from the model streaming method
        def generate_tokens():
            for token in model_instance.generate_response_stream(
                prompt=prompt,
                max_new_tokens=int(data.get("max_new_tokens", 1000)),
                temperature=float(data.get("temperature", 0.1)),
                top_p=float(data.get("top_p", 0.95)),
                repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 4)),
                do_sample=False,
                num_beams=int(data.get("num_beams", 1)),
            ):
                yield token

        # Use Flask's streaming response, yielding tokens as text/event-stream or plain text chunks
        return Response(stream_with_context(generate_tokens()), mimetype="text/plain")

    except Exception as e:
        logging.error("stream_inference", str(e))
        return jsonify(status="error", message=str(e)), 500


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


@bp_infer.route("/complete", methods=["POST"])
def complete():
    """
    Endpoint for VS Code / TabNine-like integrations.
    Expects JSON: { "context": "...", "task": "completion|bug_fix|docstring", "stream": true/false }
    """
    try:
        model_instance = load_model()

        data = request.get_json() or {}
        context = data.get("context", "").strip()
        task = data.get("task", "completion")   # default: completion
        stream = data.get("stream", False)

        if not context:
            return jsonify(status="error", message="Context is required"), 400

        # ----- Prompt engineering -----
        if task == "bug_fix":
            prompt = f"### Buggy Code:\n{context}\n### Fixed Code:\n"
        elif task == "docstring":
            prompt = f"### Code:\n{context}\n### Add a detailed docstring:\n"
        else:
            prompt = context

        logging.info(f"[Completion] Task={task}, Stream={stream}, Prompt length={len(prompt)}")

        # ---------------- STREAMING MODE ---------------- #
        if stream:
            def generate_tokens():
                try:
                    for token in model_instance.generate_response_stream(
                        prompt=prompt,
                        max_new_tokens=int(data.get("max_new_tokens", 256)),
                        temperature=float(data.get("temperature", 0.2)),
                        top_p=float(data.get("top_p", 0.95)),
                        repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                        no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                        do_sample=True,
                        num_beams=int(data.get("num_beams", 1)),
                    ):
                        yield token
                except Exception as e:
                    logging.error(f"[Stream Error] {str(e)}")
                    yield f"\n[Error] {str(e)}"

            return Response(stream_with_context(generate_tokens()), mimetype="text/plain")

        # ---------------- ONE-SHOT MODE ---------------- #
        else:
            completion = model_instance.generate_response(
                prompt=prompt,
                max_new_tokens=int(data.get("max_new_tokens", 256)),
                temperature=float(data.get("temperature", 0.2)),
                top_p=float(data.get("top_p", 0.95)),
                repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                do_sample=True,
                num_beams=int(data.get("num_beams", 1)),
            )
            logging.info(f"[Completion] Generated {len(completion)} tokens")
            logging.info(f"[Completion] Content: {completion}")
            return jsonify(status="success", completion=completion)

    except Exception as e:
        logging.error(f"[Error in /complete] {str(e)}", exc_info=True)
        return jsonify(status="error", message=str(e)), 500