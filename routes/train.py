import os
import threading
from flask import Blueprint, request, jsonify, current_app ,render_template
from models.ModelTrainer import ModelTrainer
from sql.CheckpointTrackMaster import CheckpointTrackMaster

bp_train = Blueprint("train", __name__)
training_logs = []  # In-memory training logs

def run_training(isSQL=False,max_steps=20,per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-4,num_train_epochs=10,logging_steps=2,eval_steps=5,warmup_steps=5,save_steps=20,save_strategy="steps",save_total_limit=1,metric_for_best_model="loss",fp16=True,bf16=False,greater_is_better=False,optim="adamw_torch",report_to=[],logging_dir=None):
    global training_logs
    training_logs = []

    try:
        trainer = ModelTrainer(
            is_sql=isSQL,
        )

        training_results = trainer.train(max_steps,per_device_train_batch_size,gradient_accumulation_steps,learning_rate,num_train_epochs,logging_steps,eval_steps,warmup_steps,save_steps,save_strategy,save_total_limit,metric_for_best_model,fp16,bf16,greater_is_better,optim,report_to,logging_dir)

        # Save checkpoint info to DB
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
    except Exception as e:
        training_logs.append(f"❌ Training failed: {str(e)}")


@bp_train.route("/train", methods=["POST"])
def start_training():
    data = request.get_json()
    if not data:
        return jsonify(error="No training parameters provided."), 400
    max = data.get("max_steps", 20)
    per_device_train_batch_size = data.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = data.get("gradient_accumulation_steps", 4)
    learning_rate = data.get("learning_rate", 1e-4)
    num_train_epochs = data.get("num_train_epochs", 3)
    logging_steps = data.get("logging_steps", 5)
    eval_steps = data.get("eval_steps", 50)
    warmup_steps = data.get("warmup_steps", 5)
    save_steps = data.get("save_steps", 200)
    save_strategy = data.get("save_strategy", "steps")
    save_total_limit = data.get("save_total_limit", 2)
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

    return jsonify(status="started", message="Training has been started.")


@bp_train.route("/train/logs", methods=["GET"])
def get_training_logs():
    return jsonify(logs=training_logs)


@bp_train.route("/train", methods=["GET"])
def renderTrainScreen():

    return render_template("train.html", logs=training_logs)