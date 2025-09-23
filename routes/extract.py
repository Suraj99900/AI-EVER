# routes/extract.py
import os
import threading
import tempfile
import shutil
from pathlib import Path

from flask import (
    Blueprint, request, jsonify, send_file,
    render_template, current_app
)

from models.log_manager import LogManager
from models.CodeExtractor import CodeExtractor
from models.DBSchemaExtractor import DBSchemaExtractor
from sql.AIEverLog import AIEverLog

bp_extract = Blueprint("extract", __name__)

# --- Log manager -----------------------------------------------------------
logmgr = LogManager()
logger = logmgr.setup_logger("extract")   # ensure extract logger exists


def _append_extract(msg, level=None):
    """Convenience helper to append log lines."""
    if level:
        logmgr.append("extract", str(msg), level=level)
    else:
        logmgr.append("extract", str(msg))


# --- Worker functions ------------------------------------------------------
def run_code_extraction(path: str, output_file: str, clear_logs: bool = True):
    if clear_logs:
        logmgr.clear("extract")

    _append_extract(f"Starting code extraction from: {path}")
    _append_extract(f"Output will be saved to: {output_file}")

    try:
        extractor = CodeExtractor()

        def cb(line: str):
            if line:
                logmgr.append("extract", str(line).rstrip())

        extractor.extract_code(path, output_file, callback=cb)

        _append_extract("✅ Code extraction finished. Download at /extract/download?type=code")
    except Exception as e:
        _append_extract(f"❌ Code extraction failed: {e}")
        AIEverLog().log_error("run_code_extraction", str(e))


def run_code_extraction_with_clone(repo_url: str, output_file: str,
                                   app_root: Path, clear_logs: bool = True):
    """
    Clone a GitHub repo into data/processed/repo_extractor/<temp>,
    run code extraction, then remove the clone.
    """
    if clear_logs:
        logmgr.clear("extract")

    base_clone_dir = app_root / "data" / "processed" / "repo_extractor"
    base_clone_dir.mkdir(parents=True, exist_ok=True)

    temp_clone = Path(tempfile.mkdtemp(prefix="repo_", dir=base_clone_dir))
    _append_extract(f"Cloning repository {repo_url} into {temp_clone}")

    try:
        extractor = CodeExtractor()
        extractor.clone_repo(repo_url, dest=temp_clone)
        run_code_extraction(str(temp_clone), output_file, clear_logs=False)
        _append_extract(f"✅ Extraction done. Cleaning up {temp_clone}")
    except Exception as e:
        _append_extract(f"❌ Code extraction failed: {e}")
        AIEverLog().log_error("run_code_extraction_with_clone", str(e))
    finally:
        try:
            shutil.rmtree(temp_clone)
            _append_extract(f"🧹 Removed temporary clone: {temp_clone}")
        except Exception as cleanup_err:
            _append_extract(f"⚠️ Cleanup failed: {cleanup_err}")


def run_code_extraction_ctx(app, path, output_file, clear_logs=True):
    with app.app_context():
        run_code_extraction(path, output_file, clear_logs)


def run_code_extraction_with_clone_ctx(app, repo_url, output_file, clear_logs=True):
    with app.app_context():
        run_code_extraction_with_clone(
            repo_url, output_file, Path(app.root_path), clear_logs
        )


def run_sql_extraction(host, port, user, password, db_name, clear_logs: bool = True):
    if clear_logs:
        logmgr.clear("extract")

    _append_extract("Starting SQL schema extraction")
    _append_extract(f"Connecting to MySQL: {host}:{port}, user: {user}, db: {db_name}")
    try:
        extractor = DBSchemaExtractor(host, port, user, password, db_name)

        def cb(line: str):
            if line:
                logmgr.append("extract", str(line).rstrip())

        extractor.run(callback=cb)

        _append_extract("✅ SQL schema extraction finished. Download at /extract/download?type=sql")
    except Exception as e:
        _append_extract(f"❌ SQL extraction failed: {e}")
        AIEverLog().log_error("run_sql_extraction", str(e))


# --- Routes ----------------------------------------------------------------
@bp_extract.route("/extract", methods=["POST", "GET"])
def extract():
    if request.method == "POST":
        extract_type = request.form.get("type", "code").lower()
        if extract_type not in ("code", "sql"):
            return jsonify(status="error",
                           message="❌ Invalid extract type. Use 'code' or 'sql'"), 400

        app = current_app._get_current_object()
        output_file = os.path.abspath(
            os.path.join(
                app.root_path,
                "data", "processed",
                "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl",
            )
        )
        _append_extract(f"Output file will be: {output_file}")

        if extract_type == "code":
            source_type = request.form.get("source_type", "local").lower()
            repo_path = request.form.get("repo_path", "").strip()

            if source_type == "github":
                if not repo_path:
                    return jsonify(status="error",
                                   message="❌ GitHub URL required"), 400
                thread = threading.Thread(
                    target=run_code_extraction_with_clone_ctx,
                    args=(app, repo_path, output_file, True),
                    daemon=True,
                )
            else:  # Local path
                if not repo_path or not os.path.isdir(repo_path):
                    return jsonify(status="error",
                                   message="❌ Invalid project path"), 400
                thread = threading.Thread(
                    target=run_code_extraction_ctx,
                    args=(app, repo_path, output_file, True),
                    daemon=True,
                )

        else:  # SQL branch
            host = request.form.get("host", "")
            try:
                port = int(request.form.get("port", 3306))
            except ValueError:
                port = 3306
            user = request.form.get("username", "")
            password = request.form.get("password", "")
            db_name = request.form.get("database", "")
            thread = threading.Thread(
                target=run_sql_extraction,
                args=(host, port, user, password, db_name, True),
                daemon=True,
            )

        thread.start()
        return jsonify(status="started",
                       message=f"{extract_type.capitalize()} extraction started")

    # GET -> render UI page
    return render_template("extract.html")


@bp_extract.route("/extract/logs")
def extract_logs_endpoint():
    try:
        lines = int(request.args.get("lines", 20))
    except ValueError:
        lines = 20
    return jsonify(logmgr.get_logs_json("extract", lines=lines, filter_noise=True))


@bp_extract.route("/extract/download", methods=["GET"])
def extract_downloads():
    extract_type = request.args.get("type", "code")
    filename = "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl"
    output_file = os.path.abspath(
        os.path.join(current_app.root_path, "data", "processed", filename)
    )
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404
