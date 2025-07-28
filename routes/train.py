import os
import threading
from flask import Blueprint, request, jsonify, current_app ,render_template
from models.ModelTrainer import ModelTrainer
from sql.CheckpointTrackMaster import CheckpointTrackMaster

bp_train = Blueprint("train", __name__)
training_logs = []  # In-memory training logs

def run_training(repo_path, model_dir, checkpoint_dir):
    global training_logs
    training_logs = []

    try:
        trainer = ModelTrainer(
            repo_path=repo_path,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir,
            log_callback=lambda log: training_logs.append(log)
        )

        training_results = trainer.train()

        # Save checkpoint info to DB
        db = CheckpointTrackMaster()
        db.add_checkpoint(
            model_name=training_results.get("model_name"),
            checkpoint_dir=checkpoint_dir,
            epoch=training_results.get("epoch"),
            train_loss=training_results.get("train_loss"),
            val_loss=training_results.get("val_loss"),
            accuracy=training_results.get("accuracy"),
        )
        training_logs.append("✅ Training completed and checkpoint saved.")
    except Exception as e:
        training_logs.append(f"❌ Training failed: {str(e)}")


@bp_train.route("/train", methods=["POST"])
def start_training():
    data = request.get_json()
    repo_path = data.get("repo_path")

    if not repo_path or not os.path.isdir(repo_path):
        return jsonify(status="error", message="Invalid repository path"), 400

    model_dir = os.path.join(current_app.root_path, "model", "checkpoints")
    checkpoint_dir = os.path.join(model_dir, f"checkpoint-{int(threading.get_ident())}")

    # Start training in a separate thread
    thread = threading.Thread(target=run_training, args=(repo_path, model_dir, checkpoint_dir), daemon=True)
    thread.start()

    return jsonify(status="started", message="Training has been started.")


@bp_train.route("/train/logs", methods=["GET"])
def get_training_logs():
    return jsonify(logs=training_logs)


@bp_train.route("/train", methods=["GET"])
def renderTrainScreen():

    return render_template("train.html", logs=training_logs)