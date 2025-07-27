import os
import threading
import subprocess
from flask import (
    Blueprint, render_template,
    request, jsonify, send_file, current_app
)
import sys  # Add at the top

from training import start_training
from inference import run_inference

bp = Blueprint("main", __name__, template_folder="templates")

# In-memory log buffer for extraction process
extract_logs = []

def run_extraction(path: str, scripts_dir: str, output_file: str, clear_logs: bool = True):
    """
    Run the code-extraction script asynchronously, capturing logs.
    """
    global extract_logs
    if clear_logs:
        extract_logs = []

    print(f"Starting extraction for path: {path}")
    print(f"Using scripts directory: {scripts_dir}")
    print(f"Output file will be: {output_file}")

    cmd = [
        sys.executable,
        os.path.join(scripts_dir, "extract_code.py"),
        str(path)  # pass as positional arg (remove --path if your script doesn’t support it)
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            extract_logs.append(line.rstrip())
        proc.wait()
        extract_logs.append("\n✅ Extraction finished. Download at /extract/download")
    except Exception as e:
        extract_logs.append(f"❌ Extraction failed: {str(e)}")

@bp.route("/")
def home():
    return render_template("index.html")


@bp.route("/extract", methods=["GET", "POST"])
def extract():
    if request.method == "POST":
        repo_path = request.form.get("repo_path", "")
        if not repo_path or not os.path.isdir(repo_path):
            return jsonify(status="error", message="Invalid project path"), 400

        # Capture app context values BEFORE entering the thread
        scripts_dir = os.path.join(current_app.root_path, "scripts")
        output_file = os.path.abspath(
            os.path.join(current_app.root_path, "..", "data", "processed", "train_data.jsonl")
        )

        # Pass those values into the thread safely
        thread = threading.Thread(
            target=run_extraction,
            args=(repo_path, scripts_dir, output_file, True),
            daemon=True
        )
        thread.start()
        return jsonify(status="started")

    return render_template("extract.html")



@bp.route("/extract/logs")
def extract_logs_endpoint():
    """Return live extraction logs as JSON."""
    return jsonify(logs=extract_logs)


@bp.route("/extract/download")
def extract_download():
    """Serve the generated JSONL file for download."""
    output_file = os.path.abspath(
        os.path.join(current_app.root_path, "data", "processed", "train_data.jsonl")
    )
    
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return ("File not found", 404)


@bp.route("/train", methods=["GET", "POST"])
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

@bp.route("/inference", methods=["GET", "POST"])
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
