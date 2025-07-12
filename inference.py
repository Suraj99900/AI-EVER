import subprocess
import json

def run_inference(prompt, params):
    """
    Calls scripts/inference.py and returns its stdout as JSON.
    """
    # Build CLI args
    cmd = [
        "python3", "scripts/inference.py",
        "--prompt", prompt,
        "--max_new_tokens", str(params["max_new_tokens"]),
        "--temperature", str(params["temperature"]),
        "--top_p", str(params["top_p"])
    ]
    # capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # strip any extra whitespace
    text = result.stdout.strip()
    return {"generated": text}
