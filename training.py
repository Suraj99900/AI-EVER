import threading, subprocess, os

def _train_worker(config):
    """
    This runs your scripts/train.py in a separate thread.
    You can pass config via env or CLI flags.
    """
    env = os.environ.copy()
    env["EPOCHS"] = str(config.get("epochs", 3))
    env["BATCH_SIZE"] = str(config.get("batch_size", 1))
    # … more env vars …

    # call your existing train script
    subprocess.run(
        ["python3", "scripts/train.py"],
        check=True,
        env=env
    )

def start_training(config):
    """
    Spawns a background thread to kick off training.
    Returns immediately with a status.
    """
    thread = threading.Thread(target=_train_worker, args=(config,))
    thread.daemon = True
    thread.start()
    return {"message": "Training started", "config": config}
