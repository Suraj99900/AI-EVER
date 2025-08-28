import os
import threading
from flask import Blueprint, request, jsonify, current_app ,render_template
from models.ModelTrainer import ModelTrainer
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from pathlib import Path
from models.log_manager import LogManager
import functools
import re


bp_train = Blueprint("train", __name__)
training_logs = []  # In-memory training logs

LOG_PATH = Path(__file__).parent.parent /"log/model_trainer.log"

logmgr = LogManager()
logger = logmgr.setup_logger("model_trainer")
logmgr.clear("model_trainer")

# --- Helpers ---------------------------------------------------------------
def _append_extract(msg, level=None):
    """Helper wrapper to append logs to the 'model_trainer' logger."""
    if level is None:
        logmgr.append("model_trainer", str(msg))
    else:
        logmgr.append("model_trainer", str(msg), level=level)


# Regex that matches common access log lines like:
# 127.0.0.1 - - [27/Aug/2025 21:09:02] "GET /train/logs HTTP/1.1" 200 -
_ACCESS_LOG_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\s+-\s+-\s+\[.*?\]\s+"[A-Z]+\s+([^"]+)\s+HTTP/[\d.]+"\s+\d{3}\s+.*$')

def _is_ignored_log_line(line: str) -> bool:
    """Return True if a log line is an access/werkzeug style line or otherwise noise we want to skip."""
    if not line:
        return True
    # Skip access log pattern (IP - - [time] "METHOD path HTTP/...") etc.
    if _ACCESS_LOG_RE.match(line):
        return True
    # Skip any direct polling requests to endpoints we don't want to surface in UI
    # (add more endpoints here if needed)
    if "/train/logs" in line or 'GET /train/logs' in line or 'POST /train' in line:
        return True
    # Skip typical server health checks or very short noise lines if required:
    if line.strip().startswith('127.') and 'GET' in line:
        return True
    return False

def run_training(isSQL=False,max_steps=20,per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-4,num_train_epochs=10,logging_steps=2,eval_steps=5,warmup_steps=5,save_steps=20,save_strategy="steps",save_total_limit=1,metric_for_best_model="loss",fp16=True,bf16=False,greater_is_better=False,optim="adamw_torch",report_to=[],logging_dir=None):
    global training_logs
    training_logs = []

    try:
        trainer = ModelTrainer(
            is_sql=isSQL,
        )

        training_results = trainer.train(max_steps,per_device_train_batch_size,gradient_accumulation_steps,learning_rate,num_train_epochs,logging_steps,eval_steps,warmup_steps,save_steps,save_strategy,save_total_limit,metric_for_best_model,fp16,bf16,greater_is_better,optim,report_to,logging_dir)

        # Save checkpoint info to DB
        _append_extract("Saving training checkpoint to database.")
        db = CheckpointTrackMaster()
        db.add_checkpoint(
            model_name=training_results.get("model_name"),
            checkpoint_dir=training_results.get("checkpoint_dir"),
            epoch=training_results.get("epoch"),
            train_loss=training_results.get("train_loss"),
            val_loss=training_results.get("val_loss"),
            accuracy=training_results.get("accuracy"),
        )
        training_logs.append("✅ Training completed and checkpoint saved.")
        _append_extract("✅ Training completed and checkpoint saved.")
    except Exception as e:
        training_logs.append(f"❌ Training failed: {str(e)}")
        _append_extract(f"❌ Training failed: {str(e)}")


@bp_train.route("/train", methods=["POST"])
def start_training():
    data = request.get_json()
    if not data:
        return jsonify(error="No training parameters provided."), 400
    max = int(data.get("max_steps", 20))
    per_device_train_batch_size = int(data.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(data.get("gradient_accumulation_steps", 4))
    learning_rate = data.get("learning_rate", 1e-4)
    num_train_epochs = int(data.get("num_train_epochs", 3))
    logging_steps = int(data.get("logging_steps", 5))
    eval_steps = int(data.get("eval_steps", 50))
    warmup_steps = int(data.get("warmup_steps", 5))
    save_steps = int(data.get("save_steps", 200))
    save_strategy = data.get("save_strategy", "steps")
    save_total_limit = int(data.get("save_total_limit", 2))
    metric_for_best_model = data.get("metric_for_best_model", "loss")
    fp16 = data.get("fp16", True)
    bf16 = data.get("bf16", False)
    greater_is_better = data.get("greater_is_better", False)
    optim = data.get("optim", "adamw_torch")
    report_to = data.get("report_to", [])
    logging_dir = data.get("logging_dir", None)
    isSQL = data.get("is_sql", False)

    # Start training in a separate thread
    thread = threading.Thread(target=run_training, args=(isSQL,max,per_device_train_batch_size,gradient_accumulation_steps,learning_rate,num_train_epochs,logging_steps,eval_steps,warmup_steps,save_steps,save_strategy,save_total_limit,metric_for_best_model,fp16,bf16,greater_is_better,optim,report_to,logging_dir), daemon=True)
    thread.start()
    _append_extract("Training started in background thread.")

    return jsonify(status="started", message="Training has been started.")


@bp_train.route("/train/logs", methods=["GET"])
def get_training_logs():
    """
    Return the last N meaningful lines from the log file, but skip access/werkzeug lines
    and other noisy entries (like polling requests).
    """
    lines = []
    N = int(request.args.get("lines", 1))  # Number of non-ignored lines to return

    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "rb") as f:
                f.seek(0, os.SEEK_END)
                pointer = f.tell() - 1
                buffer = bytearray()
                collected = []

                # Walk file backwards until we have N non-ignored lines or reach start
                while pointer >= 0 and len(collected) < N:
                    f.seek(pointer)
                    b = f.read(1)
                    if b == b'\n':
                        # end of a line; decode reversed buffer
                        raw_line = buffer[::-1].decode(errors="ignore").strip()
                        buffer = bytearray()
                        if raw_line and not _is_ignored_log_line(raw_line):
                            collected.append(raw_line)
                    else:
                        buffer.extend(b)
                    pointer -= 1

                # handle the first line (when pointer < 0) if any remaining
                if buffer:
                    raw_line = buffer[::-1].decode(errors="ignore").strip()
                    if raw_line and not _is_ignored_log_line(raw_line) and len(collected) < N:
                        collected.append(raw_line)

                # reverse to chronological order
                lines = collected[::-1]

            if not lines:
                lines = ["$ Waiting for extraction..."]
        else:
            lines = ["$ Waiting for extraction..."]
    except Exception as e:
        lines = [f"Error reading log: {e}"]

    return jsonify(logs=lines)

@bp_train.route("/train", methods=["GET"])
def renderTrainScreen():

    return render_template("train.html", logs=training_logs)