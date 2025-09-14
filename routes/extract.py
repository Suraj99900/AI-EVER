# routes/extract.py
import os
import threading
from flask import Blueprint, request, jsonify, send_file, render_template, current_app
from pathlib import Path

# your LogManager implementation (must expose: setup_logger, append, clear, get_logs_json)
from models.log_manager import LogManager
from models.CodeExtractor import CodeExtractor
from models.DBSchemaExtractor import DBSchemaExtractor
from sql.AIEverLog import AIEverLog

bp_extract = Blueprint("extract", __name__)

# LogManager singleton
logmgr = LogManager()
# ensure logger exists and is configured (creates log/extract.log etc)
logger = logmgr.setup_logger("extract")


# --- Helpers ---------------------------------------------------------------
def _append_extract(msg, level=None):
    """Helper wrapper to append to extract logger (keeps call-sites short)."""
    # IMPORTANT: LogManager.append(logger_name, message, level=...) signature assumed
    if level is None:
        logmgr.append("extract", str(msg))
    else:
        logmgr.append("extract", str(msg), level=level)


# --- Worker functions ------------------------------------------------------
def run_code_extraction(path: str, output_file: str, clear_logs: bool = True):
    """
    Runs code extraction in background. Appends progress lines to LogManager
    using the 'extract' logger so UI can poll /extract/logs.
    """
    if clear_logs:
        logmgr.clear("extract")

    _append_extract(f"Starting code extraction from: {path}")
    _append_extract(f"Output will be saved to: {output_file}")

    try:
        extractor = CodeExtractor()

        # pass a callback that writes into LogManager directly (thread-safe)
        def cb(line: str):
            # sanitize line (strip newlines) and append
            if line is None:
                return
            # Many extractors may pass bytes or objects; coerce to str
            logmgr.append("extract", str(line).rstrip())

        # call extractor
        extractor.extract_code(path, output_file, callback=cb)

        _append_extract("✅ Code extraction finished. Download at /extract/download?type=code")
    except Exception as e:
        _append_extract(f"❌ Code extraction failed: {e}")
        # Track in your DB/log table as well
        AIEverLog().log_error("run_code_extraction", str(e))


def run_sql_extraction(host, port, user, password, db_name, clear_logs: bool = True):
    if clear_logs:
        logmgr.clear("extract")

    _append_extract("Starting SQL schema extraction")
    _append_extract(f"Connecting to MySQL: {host}:{port}, user: {user}, db: {db_name}")
    try:
        extractor = DBSchemaExtractor(host, port, user, password, db_name)

        # again pass callback to capture lines
        def cb(line: str):
            if line is None:
                return
            logmgr.append("extract", str(line).rstrip())

        # If your DBSchemaExtractor.run accepts callback, pass it; else adapt accordingly
        extractor.run(callback=cb)

        _append_extract("✅ SQL schema extraction finished. Download at /extract/download?type=sql")
    except Exception as e:
        _append_extract(f"❌ SQL extraction failed: {e}")
        AIEverLog().log_error("run_sql_extraction", str(e))


# --- Routes ---------------------------------------------------------------
@bp_extract.route("/extract", methods=["GET", "POST"])
def extract():
    if request.method == "POST":
        extract_type = request.form.get("type", "code").lower()
        if extract_type not in ("code", "sql"):
            return jsonify(status="error", message="Invalid extract type. Use 'code' or 'sql'"), 400

        output_file = os.path.abspath(
            os.path.join(current_app.root_path, "data", "processed",
                         "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl")
        )
        _append_extract(f"Output file will be: {output_file}")

        if extract_type == "code":
            repo_path = request.form.get("repo_path", "")
            if not repo_path or not os.path.isdir(repo_path):
                return jsonify(status="error", message="Invalid project path"), 400

            thread = threading.Thread(target=run_code_extraction, args=(repo_path, output_file, True), daemon=True)

        else:
            host = request.form.get("host", "")
            try:
                port = int(request.form.get("port", 3306))
            except Exception:
                port = 3306
            user = request.form.get("username", "")
            password = request.form.get("password", "")
            db_name = request.form.get("database", "")

            thread = threading.Thread(
                target=run_sql_extraction,
                args=(host, port, user, password, db_name, True),
                daemon=True
            )

        thread.start()
        return jsonify(status="started", message=f"{extract_type.capitalize()} extraction started")

    # GET -> render page
    return render_template("extract.html")


@bp_extract.route("/extract/logs")
def extract_logs_endpoint():
    """
    Return JSON with last N lines using LogManager (preferred).
    The LogManager does in-memory deque fallback and file tailing.
    Avoid reading the raw file directly here.
    """
    try:
        lines = int(request.args.get("lines", 20))
    except Exception:
        lines = 20

    # filter_noise=True will drop HTTP access-log lines and other noisy progressbar lines
    return jsonify(logmgr.get_logs_json("extract", lines=lines, filter_noise=True))


@bp_extract.route("/extract/download", methods=["GET"])
def extract_downloads():
    extract_type = request.args.get("type", "code")
    filename = "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl"
    output_file = os.path.abspath(os.path.join(current_app.root_path, "data", "processed", filename))
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return ("File not found", 404)
