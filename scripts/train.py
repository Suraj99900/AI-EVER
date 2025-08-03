#!/usr/bin/env python3
import os, logging, math, torch, datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────
MODEL_DIR  = "../model/deepseek-coder-1.3b-base"
# DATA_PATH  = "../data/processed/train_data.jsonl"
DATA_PATH  = "../data/content/datasets/extracted_python_code.jsonl"

# Create timestamped checkpoint directory for uniqueness
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../model/checkpoints/deepseek-1.3b-lora-{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device & Quant Config ───────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ── Tokenizer ───────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load & Prep Model ───────────────────────────────────
logger.info("Loading base model in 4‑bit…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model = prepare_model_for_kbit_training(base_model)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_cfg)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(device)

# ── Data ────────────────────────────────────────────────
full = load_dataset("json", data_files=DATA_PATH, split="train")
splits = full.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds  = splits["test"]

def tokenize_fn(batch):
    toks = tokenizer(
        batch["text"], 
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    input_ids = toks["input_ids"]
    labels = [
        [tok if tok != tokenizer.pad_token_id else -100 for tok in seq]
        for seq in input_ids
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": toks["attention_mask"],
        "labels": labels
        }

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map (tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False, 
    pad_to_multiple_of=8
)

# ── Metrics ─────────────────────────────────────────────
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
    return {"perplexity": perplexity}

# ── TrainingArgs ────────────────────────────────────────
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=12,
#     learning_rate=1e-4,
#     num_train_epochs=5,
#     logging_steps=10,
#     eval_steps=200,
#     warmup_steps=300,
#     save_steps=200,
#     save_strategy="steps",
#     save_total_limit=3,
#     metric_for_best_model="perplexity",
#     fp16=torch.cuda.is_available(),
#     bf16=False,
#     optim="adamw_torch",
#     report_to=[],
#     logging_dir=os.path.join(OUTPUT_DIR, "logs"),
# )

# For Large Datasets, Reduce Epochs
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     learning_rate=2e-4,
#     num_train_epochs=1,                      # Reduced due to small dataset
#     logging_steps=10,
#     eval_steps=50,
#     warmup_steps=10,
#     save_steps=50,
#     save_strategy="steps",
#     save_total_limit=1,
#     metric_for_best_model="eval_loss",
#     fp16=torch.cuda.is_available(),
#     bf16=False,
#     optim="adamw_torch",
#     report_to=[],
#     logging_dir=os.path.join(OUTPUT_DIR, "logs"),
#     greater_is_better=False,
# )

# For Small Datasets, Increase Epochs
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,                      # Increased for small dataset
    logging_steps=2,
    eval_steps=50,
    warmup_steps=5,
    warmup_ratio=0.03,
    save_steps=200,
    save_strategy="steps",
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    bf16=False,
    greater_is_better=False,
    optim="adamw_torch",
    report_to=[],
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
)


# ── Resume from Last Checkpoint ─────────────────────────
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
                   if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getctime)
        logger.info(f"Resuming training from last checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logger.info("Starting training…")
trainer.train(resume_from_checkpoint=last_checkpoint)

# ── Save Final Model ────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
