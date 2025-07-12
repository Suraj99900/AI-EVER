AI Ever
A developer‑focused platform to train and infer on your own codebase and MySQL schema using LLMs (e.g. StarCoder, CodeLLaMA) with efficient fine‑tuning (QLoRA/LoRA) — all via a clean, Tailwind‑powered web UI.

🚀 Features
Automatic Code & SQL Extraction
Scan your project directory to generate instruction‑response prompts from every source file.

Schema Introspection
Extract table definitions, column details & row counts from any MySQL database.

Efficient Fine‑Tuning
QLoRA / LoRA adapters minimize GPU/CPU usage and support checkpoint resume.

One‑Click Training
Kick off fine‑tuning from the browser; monitor live logs & download progress.

Interactive Inference
Submit structured prompts (code questions, SQL queries, etc.) and get context‑aware responses.

📦 Installation
Clone repository


git clone https://github.com/yourusername/ai‑ever.git
cd ai‑ever
Create & activate Python venv


python3 -m venv .venv
source .venv/bin/activate
Install dependencies


pip install -r requirements.txt
Download base model
Place your pretrained StarCoder‑1B (or other) under model/starcoderbase-1b/.
It should contain config.json, pytorch_model.bin / model.safetensors, tokenizer.*, etc.

Configure paths
Ensure the following folders exist (or adjust in app/config.py):


data/raw_code/         ← where your repo will be cloned
data/processed/        ← where JSONL extraction lives
model/checkpoints/     ← adapter checkpoints & tokenizer
model/model_cache/     ← HF cache for quantized weights
⚙️ Usage
1. Launch the Web UI

export FLASK_APP=run.py
export FLASK_ENV=development
flask run
Visit http://127.0.0.1:5000 in your browser.

2. Extract Code & Schema
Code Extraction:
Go to Extract Code, enter your local project path (e.g. /home/user/WebApp-Loan), click Start Extraction.
Live logs will appear; click Download JSONL once complete.

DB Extraction:
(if enabled) similarly extract your MySQL schema by entering connection details.

3. Start Training
Click Train in the navbar.

Fill in hyperparameters (epochs, batch size, learning rate) and click Launch Training.

Live logs stream in the text box.

On completion, your LoRA adapter weights and tokenizer will be saved under model/checkpoints/.

Resume training
If you restart the same training job, the app will detect trainer_state.json in model/checkpoints/ and resume automatically.
To start fresh, change your output directory in the form or delete the old checkpoints.

4. Run Inference
Click Inference

Enter a prompt in the form:


### Instruction:
Write an SQL query to list active users.

### Response:
Click Generate and see the model’s output below.

🛠️ Project Structure
pgsql

ai-ever/
├── app/
│   ├── __init__.py        # Flask factory
│   ├── routes.py          # All UI routes & extraction endpoints
│   ├── training.py        # start_training() wrapper
│   ├── inference.py       # run_inference() wrapper
│   └── templates/
│       ├── base.html      # Glass‑morphic layout
│       ├── index.html     # Landing page
│       ├── extract.html   # Code extraction UI
│       ├── train.html     # Training UI
│       └── inference.html # Inference UI
├── scripts/
│   ├── extract_code.py    # Walks RAW_DIR → produces JSONL
│   ├── extract_db_info.py # Dumps MySQL schema → JSON/text
│   └── train.py           # Fine‑tune with QLoRA/LoRA
├── data/
│   ├── raw_code/          # Cloned repositories
│   └── processed/         # `train_data.jsonl`, `db_summary.json`
├── model/
│   ├── starcoderbase-1b/  # Pretrained HF model
│   ├── model_cache/       # HF quantized cache
│   └── checkpoints/       # Saved LoRA adapters & tokenizer
├── requirements.txt
├── run.py                 # Flask app entrypoint
└── README.md
🙋‍♂️ Troubleshooting
OutOfMemoryError

Make sure you’re using 4‑bit / 8‑bit quantization (BitsAndBytesConfig).

Increase gradient_accumulation_steps or reduce max_length.

Enable expandable_segments:


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Permission Denied / Missing Templates

Verify you ran flask run from the project root.

Ensure app/templates/ exists and contains all .html files.

MySQL Connection Fails

Check scripts/extract_db_info.py’s DB_CONFIG.

Confirm your user/password and that mysql-connector-python is installed.