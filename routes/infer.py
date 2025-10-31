from math import log
import os
from flask import Blueprint, request, jsonify, render_template, current_app,Response, stream_with_context
from transformers import pipeline

from models.ModelInference import ModelInference
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog
import logging
from pathlib import Path
import shutil
from sql.AIEverInferenceLog import AIEverInferenceLog

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
current_model_id = None

def load_model(iModelId=999):
    global model_instance, current_model_id

    if model_instance is None:
        try:
            print(f"Loading model with ID: {iModelId}- {model_instance}")
            logging.info(f"[Model Loader] Initializing ModelInference for {iModelId} ...")
            model_instance = ModelInference(iModelId=iModelId)
            current_model_id = iModelId
            print(f"Loading model with ID: {current_model_id}- {model_instance}")
            logging.info("[Model Loader] Model loaded successfully")
        except Exception as e:
            logging.error(f"[Model Loader] Failed: {e}")
            raise

    return model_instance

def get_model(iModelId=None):
    global model_instance
    if model_instance is None:
        logging.info(f"[Model] Loading model for session {iModelId}")
        model_instance = load_model(iModelId)
    return model_instance

def unload_model():
    global model_instance, current_model_id
    if model_instance is not None:
        logging.info("[Model Loader] Unloading previous model...")
        try:
            if hasattr(model_instance, "close"):
                model_instance.close()
            # Explicitly delete
            del model_instance
            model_instance = None
            current_model_id = None
            logging.info("[Model Loader] Previous model unloaded successfully")
        except Exception as e:
            logging.error(f"[Model Loader] Failed to unload model: {e}")

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
    log_db = AIEverInferenceLog()
    CHECKPOINTS = fetch_checkpoints()

    if request.method == "POST":
        # --- existing POST logic for new prompts ---
        data = request.get_json() or {}
        iModelId = data.get("iModelId", 999)
        load_model(iModelId)
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify(status="error", message="Prompt is required"), 400

        # Format the prompt for the model
        full_prompt = f"### Instruction:\n{prompt}\n### Response:"
        logging.info(f"Received prompt: {full_prompt}")

        try:
            result = model_instance.generate_response_stream(
                prompt=full_prompt,
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

    else:
        # --- GET logic: render the chat page ---
        iModelId = request.args.get("session", default=9999, type=int)
        load_model(iModelId)

        base = os.getenv("BASE_URL", "http://localhost:5000").rstrip("/")
        BASE_URL = f"{base}/inference"

        # Format checkpoints for the template
        formatted_checkpoints = []
        seen = set()
        sBaseModel = Path(__file__).parent.parent / "LLMModels" / "deepseek-coder-1.3b-base"
        formatted_checkpoints.append((9999, "deepseek-coder-1.3b-base", str(sBaseModel)))

        for cp in CHECKPOINTS:
            if not os.path.exists(cp[2]):
                continue
            folder_name = os.path.basename(cp[2])
            if folder_name not in seen:
                seen.add(folder_name)
                formatted_checkpoints.append((cp[0], folder_name, cp[2]))

        # --- Fetch past logs for this model/session ---
        past_chats = log_db.get_logs_by_checkpoint(iModelId)  # Returns list of dicts: [{req_msg, res_msg}, ...]

        return render_template(
            "inference.html",
            checkpoints=formatted_checkpoints,
            current_session=iModelId,
            BASE_URL=BASE_URL,
            past_chats=past_chats
        )

@bp_infer.route("/stream_inference", methods=["POST"])
def stream_inference():
    iModelId = request.args.get('current_session', default=9999, type=int)
    load_model(iModelId)

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify(status="error", message="Prompt is required"), 400

    # Format the prompt for the model
    full_prompt = f"### Instruction:\n{prompt}\n### Response:"
    logging.info(f"Received streaming prompt: {full_prompt}")

    log_db = AIEverInferenceLog()
    log_id = log_db.add_log(
        check_point_id=iModelId,
        req_msg=full_prompt,
        res_msg="",
        related_checkpoint_id=iModelId
    )

    try:
        def generate_tokens_and_log():
            response_buffer = []
            try:
                for token in model_instance.generate_response_stream(
                    prompt=full_prompt,
                    max_new_tokens=int(data.get("max_new_tokens", 1000)),
                    temperature=float(data.get("temperature", 0.1)),
                    top_p=float(data.get("top_p", 0.95)),
                    repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                    no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 4)),
                    do_sample=False,
                    num_beams=int(data.get("num_beams", 1)),
                ):
                    response_buffer.append(token)
                    yield token
            finally:
                try:
                    final_response = "".join(response_buffer)
                    log_db.update_log(log_id, res_msg=final_response)
                except Exception as db_err:
                    logging.error(f"[Logging Error] Could not update inference log {log_id}: {db_err}")

        # 4️⃣ Stream back to the client
        return Response(
            stream_with_context(generate_tokens_and_log()),
            mimetype="text/plain"
        )

    except Exception as e:
        logging.error("stream_inference", str(e))
        if log_id:
            log_db.update_log(log_id, status=0, res_msg=f"[Error] {str(e)}")
        return jsonify(status="error", message=str(e)), 500


@bp_infer.route("/rename_checkpoint/<int:cp_id>", methods=["POST"])
def rename_checkpoint(cp_id):
    data = request.get_json(silent=True) or {}
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"success": False, "error": "Invalid name"}), 400

    try:
        db = CheckpointTrackMaster()
        cp = db.get_checkpoint_by_id(cp_id)
        if not cp:
            return jsonify({"success": False, "error": "Checkpoint not found"}), 404
        
        old_path = cp[2]        # assuming column 2 = file path
        base_dir = os.path.dirname(old_path)
        
        new_path = os.path.join(base_dir, new_name)
        print(f"new_path {new_path}")

        # # rename file on disk
        os.rename(old_path, new_path)

        # # update database
        db.update_checkpoint(cp_id, checkpoint_dir=new_path)

        return jsonify({"success": True})
    except Exception as e:
        # Log full traceback if you have logging configured
        return jsonify({"success": False, "error": str(e)}), 500


@bp_infer.route("/delete_checkpoint/<int:cp_id>", methods=["DELETE"])
def delete_checkpoint(cp_id):
    data = request.get_json(silent=True) or {}
    if not cp_id:
        return jsonify({"success": False, "error": "Invalid checkpoint ID"}), 400

    try:
        db = CheckpointTrackMaster()
        cp = db.get_checkpoint_by_id(cp_id)
        if not cp:
            return jsonify({"success": False, "error": "Checkpoint not found"}), 404

        path_to_delete = cp[2]  # assuming column 2 = file path

        # delete folder and its contents
        if os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)   # <— removes all files & subdirs

        # remove the DB record
        db.delete_checkpoint(cp_id)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

@bp_infer.route("/complete", methods=["POST"])
def complete():
    """
    Endpoint for VS Code / TabNine-like integrations.
    Expects JSON: { "context": "...", "task": "completion|bug_fix|docstring", "stream": true/false }
    """
    try:
        iModelId = request.args.get('current_session', default=9999, type=int)
        model = get_model(iModelId)

        data = request.get_json() or {}
        context = data.get("context", "").strip()
        task = data.get("task", "completion")
        stream = data.get("stream", False)

        if not context:
            return jsonify(status="error", message="Context is required"), 400

        # ----- Prompt engineering -----
        if task == "bug_fix":
            prompt = f"### Buggy Code:\n{context}\n### Fixed Code:\n"
        elif task == "docstring":
            prompt = f"### Code:\n{context}\n### Add a detailed docstring:\n"
        elif task == "completion":
            prompt = f"### complete this code:\n{context}\n"
        else:
            prompt = context

        logging.info(f"[Completion] Task={task}, Stream={stream}, Prompt length={len(prompt)}")
        logging.info(prompt)

        # ---------------- STREAMING MODE ---------------- #
        if stream:
            def generate_tokens():
                try:
                    for token in model.generate_response_stream(
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
            completion = model.generate_response(
                prompt=prompt,
                max_new_tokens=int(data.get("max_new_tokens", 256)),
                temperature=float(data.get("temperature", 0.2)),
                top_p=float(data.get("top_p", 0.95)),
                repetition_penalty=float(data.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(data.get("no_repeat_ngram_size", 3)),
                do_sample=True,
                num_beams=int(data.get("num_beams", 1)),
            )

            if not completion:
                logging.warning("[Completion] Model returned empty string")
                return jsonify(status="error", message="Empty completion"), 500

            logging.info(f"[Completion] Generated {len(completion)} tokens")
            return jsonify(status="success", completion=completion)

    except Exception as e:
        logging.error(f"[Error in /complete] {str(e)}", exc_info=True)
        return jsonify(status="error", message=str(e)), 500
    

@bp_infer.route("/switch_model")
def switch_model():
    global model_instance, current_model_id
    new_id = request.args.get("id", type=int)
    # If we already have a model but a different ID is requested, unload first
    if model_instance is not None and new_id != current_model_id:
        logging.info(f"[Model Loader] Switching model: {current_model_id} -> {new_id}")
        unload_model()
        load_model(new_id)
    return jsonify(status="success", current_model=new_id)