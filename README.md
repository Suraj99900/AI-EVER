# AI-EVER

AI-EVER is a developer-focused platform that lets you **train and run large language models on your own codebase and database schema**.  
It extracts source code and MySQL table structures, creates instruction-response prompts, and fine-tunes an open-source model (e.g. CodeLlama, StarCoder, DeepSeek-Coder) using LoRA/QLoRA—all from a web UI built with Flask and Tailwind CSS.

---

## ✨ Key Features
* **Code & Schema Extraction** – Scan any project directory and optional MySQL database to build a JSONL training set.
* **Low-Resource Fine-Tuning** – LoRA / QLoRA adapters for GPUs with as little as 4 GB VRAM.
* **Checkpoint Management** – Resume or rename checkpoints directly from the browser.
* **Interactive Inference** – Ask questions about your codebase and get context-aware answers.

---

## 🛠️ Prerequisites
* **Python** 3.10+  
* **pip** and **virtualenv**  
* Optional: **CUDA-capable GPU** (recommended for training)  
* MySQL server if you plan to extract database schema

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-ever.git
cd ai-ever


2️⃣ Create & Activate a Virtual Environment

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or on Windows:
# .venv\Scripts\activate



3️⃣ Install Python Dependencies

pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Prepare Model Files

Download a base model (e.g. deepseek-coder-1.3b-base, CodeLlama, or StarCoder) and place it under:

LLMModels/<model-name>/


Required files: config.json, tokenizer.*, and model weights (pytorch_model.bin or model.safetensors).

5️⃣ Create Data & Checkpoint Folders

If they don’t already exist:

data/raw_code/        # your source projects
data/processed/       # generated JSONL datasets
LLMModels/checkpoints # training checkpoints
LLMModels/model_cache # HF cache for quantized weights


▶️ Running the Web App

Set environment variables (optional)
You can override defaults in .env or your shell:

export FLASK_APP=run.py
export FLASK_ENV=development

Start the server
flask run

Open http://127.0.0.1:5000    in your browser.


🧩 Typical Workflow

Extract Code / Schema

Go to Extract Code in the UI.

Enter the local project path (and optional MySQL credentials).

Click Start Extraction to generate train_data.jsonl.

Train a Model

Navigate to Train.

Fill in hyperparameters (epochs, learning rate, etc.).

Click Launch Training.

Checkpoints appear in LLMModels/checkpoints/ and can be resumed later.

Inference

Open Inference.

Choose a checkpoint or the base model.

Enter a prompt (e.g. “Write a SQL query to list active users”) and click Generate.

⚙️ Tips & Troubleshooting

Low GPU Memory – Use 4-bit or 8-bit quantization and increase gradient_accumulation_steps.

MySQL Issues – Ensure mysql-connector-python is installed and credentials are correct.

Template Errors – Run flask run from the project root so Flask finds the app/templates/ directory.



ai-ever/
├── app/
│   ├── __init__.py
│   ├── routes/        # Flask blueprints
│   ├── training.py
│   ├── inference.py
│   └── templates/
├── data/
│   ├── raw_code/
│   └── processed/
├── LLMModels/
│   ├── deepseek-coder-1.3b-base/
│   ├── checkpoints/
│   └── model_cache/
├── scripts/
│   ├── extract_code.py
│   ├── extract_db_info.py
│   └── train.py
├── requirements.txt
├── run.py
└── README.md



---

### How to Add This File
Save the above as `README.md` in your project root, then commit:

```bash
git add README.md
git commit -m "Add comprehensive README with setup & usage instructions"
git push origin main
