import os
import sys
import threading
from flask import Blueprint, request, jsonify, send_file, render_template, current_app

from models.CodeExtractor import CodeExtractor
from models.ModelTrainer import ModelTrainer
from models.ModelInference import ModelInference
from models.DBSchemaExtractor import DBSchemaExtractor
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog

bp_extract = Blueprint("extract", __name__)

# In-memory log buffer
extract_logs = []


def run_code_extraction(path: str, output_file: str, clear_logs: bool = True):
    global extract_logs
    if clear_logs:
        extract_logs = []
    print(f"Starting code extraction from: {path}")
    extract_logs.append(f"Starting code extraction from: {path}")
    extract_logs.append(f"Output will be saved to: {output_file}")
    try:
        extractor = CodeExtractor()
        oResult = extractor.extract_code(
            path,
            output_file,
            callback=lambda line: extract_logs.append(line)
        )
        extract_logs.append("\n✅ Code extraction finished. Download at /extract/download?type=code")
    except Exception as e:
        extract_logs.append(f"❌ Code extraction failed: {str(e)}")
        AIEverLog().log_error("run_code_extraction", str(e))
    
    return oResult


def run_sql_extraction(host, port, user, password, db_name, clear_logs: bool = True):
    global extract_logs
    if clear_logs:
        extract_logs = []

    print("Starting SQL schema extraction")
    print(f"Connecting to MySQL: {host}:{port}, user: {user}, db: {db_name}")
    extract_logs.append(f"Connecting to MySQL: {host}:{port}, user: {user}, db: {db_name}")
    extract_logs.append("Starting SQL schema extraction")
    try:
        extract_logs.append("🔌 Connecting to MySQL...")
        extractor = DBSchemaExtractor(host, port, user, password, db_name)
        extractor.run()
        extract_logs.append("\n✅ SQL schema extraction finished. Download at /extract/download?type=sql")
    except Exception as e:
        print("error in run_sql_extraction:", e)
        extract_logs.append(f"❌ SQL extraction failed: {str(e)}")
        AIEverLog().log_error("run_sql_extraction", str(e))



@bp_extract.route("/extract", methods=["GET", "POST"])
def extract():
    if request.method == "POST":
        extract_type = request.form.get("type", "code").lower()

        if extract_type not in ["code", "sql"]:
            return jsonify(status="error", message="Invalid extract type. Use 'code' or 'sql'"), 400

        output_file = os.path.abspath(os.path.join(current_app.root_path, "data", "processed",
                                                   "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl"))
        extract_logs.append(f"Output file will be: {output_file}")

        if extract_type == "code":
            repo_path = request.form.get("repo_path", "")
            if not repo_path or not os.path.isdir(repo_path):
                return jsonify(status="error", message="Invalid project path"), 400

            thread = threading.Thread(target=run_code_extraction, args=(repo_path, output_file, True), daemon=True)

        else:  # SQL extraction
            # 🟢 Extract form data and pass as args
            host = request.form.get("host", "")
            port = int(request.form.get("port", 3306))
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

    return render_template("extract.html")



@bp_extract.route("/extract/logs")
def extract_logs_endpoint():
    return jsonify(logs=extract_logs)


@bp_extract.route("/extract/download", methods=["GET"])
def extract_downloads():
    extract_type = request.args.get("type", "code")
    filename = "train_data.jsonl" if extract_type == "code" else "train_sql.jsonl"
    output_file = os.path.abspath(os.path.join(current_app.root_path, "data", "processed", filename))
    print(f"Preparing to send file: {output_file}")
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404

