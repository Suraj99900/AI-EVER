# models/ModelTrainer.py

import os
import math
import torch
import datetime
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog


class ModelTrainer:
    def __init__(
        self,
        model_dir=None,
        code_data_path=None,
        sql_data_path=None,
        base_output_path=None,
        is_sql=False
    ):
        try:
            base_folder = Path(__file__).parent
            # Resolve model & data paths
            self.model_dir = Path(model_dir or base_folder / "../LLMModels/deepseek-coder-1.3b-base").resolve()
            self.code_data_path = Path(code_data_path or base_folder / "../data/processed/train_data.jsonl").resolve()
            self.sql_data_path = Path(sql_data_path or base_folder / "../data/processed/train_sql.jsonl").resolve()
            self.data_path = self.sql_data_path if is_sql else self.code_data_path

            # Prepare output directory base
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_base = Path(base_output_path) if base_output_path else base_folder / "../LLMModels/checkpoints"
            self.output_dir = (out_base / f"{self.model_dir.name}-lora-{self.timestamp}").resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s - %(message)s"
            )
            self.logger = logging.getLogger(__name__)

            # Database interfaces
            self.checkpoint_db = CheckpointTrackMaster()
            self.log_db = AIEverLog()

            # Device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Using device: %s", self.device)
        except Exception as e:
            logging.error("Initialization failed: %s", e)
            raise

    def load_model_and_tokenizer(self):
        try:
            self.logger.info("Loading model and tokenizer from %s...", self.model_dir)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                use_fast=False,
                local_files_only=True,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
            base_model = prepare_model_for_kbit_training(base_model)

            # LoRA config
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            self.model = get_peft_model(base_model, lora_cfg)
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            self.model.to(self.device)
        except Exception as e:
            self.logger.error("Model/tokenizer loading failed: %s", e)
            raise

    def prepare_data(self):
        try:
            self.logger.info("Preparing dataset from %s...", self.data_path)
            ds = load_dataset("json", data_files=str(self.data_path), split="train")
            splits = ds.train_test_split(test_size=0.1, seed=42)
            train_ds, eval_ds = splits["train"], splits["test"]

            def tokenize_fn(batch):
                toks = self.tokenizer(
                    batch["text"], truncation=True, padding="longest", max_length=512
                )
                input_ids = toks["input_ids"]
                labels = [[tok if tok != self.tokenizer.pad_token_id else -100 for tok in seq] for seq in input_ids]
                return {"input_ids": input_ids, "attention_mask": toks["attention_mask"], "labels": labels}

            self.train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
            self.eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        except Exception as e:
            self.logger.error("Data preparation failed: %s", e)
            raise

    def compute_metrics(self, eval_preds):
        try:
            logits, labels = eval_preds
            shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            return {"perplexity": math.exp(loss.item())}
        except Exception as e:
            self.logger.error("Metric computation failed: %s", e)
            return {}
    def find_latest_checkpoint(self,base_dir):
        latest_ckpt = None
        latest_time = 0
        for run_dir in Path(base_dir).glob(f"{self.model_dir.name}-lora-*"):
            for ckpt_dir in run_dir.glob("checkpoint-*"):
                mtime = ckpt_dir.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_ckpt = ckpt_dir
        return str(latest_ckpt) if latest_ckpt else None

    def train(
        self,
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=2,
        eval_steps=50,
        warmup_steps=5,
        save_steps=200,
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),
        bf16=False,
        greater_is_better=False,
        optim="adamw_torch",
        report_to=None,
        logging_dir=None
    ):
        report_to = report_to or []
        checkpoint_id = None
        try:
            self.logger.info("Initializing training configuration...")

            # Detect existing checkpoint to resume
            base_cp_path = self.output_dir.parent

            last_checkpoint = self.find_latest_checkpoint(self.output_dir.parent)
            if last_checkpoint:
                self.logger.info(f"Resuming training from last checkpoint: {last_checkpoint}")
            else:
                self.logger.info("No checkpoint found. Starting fresh training.")
            # Create fresh run directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Output directory: %s", self.output_dir)

            # Load components
            self.load_model_and_tokenizer()
            self.prepare_data()

            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=int(per_device_train_batch_size),
                gradient_accumulation_steps=int(gradient_accumulation_steps),
                max_steps=int(max_steps),
                learning_rate=float(learning_rate),
                num_train_epochs=int(num_train_epochs),
                logging_steps=int(logging_steps),
                eval_steps=eval_steps,
                warmup_steps=int(warmup_steps),
                save_steps=int(save_steps),
                save_strategy=save_strategy,
                save_total_limit=int(save_total_limit),
                metric_for_best_model=metric_for_best_model,
                fp16=fp16,
                bf16=bf16,
                warmup_ratio = 0.03,
                greater_is_better=greater_is_better,
                optim=optim,
                report_to=report_to,
                gradient_checkpointing=True,
                ddp_find_unused_parameters =False,
                dataloader_num_workers=2,
                group_by_length=True,
                weight_decay=0.01,
                logging_dir=str(logging_dir or self.output_dir / "logs")
            )

            # Register checkpoint entry
            checkpoint_id = self.checkpoint_db.add_checkpoint(
                model_name=self.model_dir.name,
                checkpoint_dir=str(self.output_dir),
                epoch=0, train_loss=0.0, val_loss=0.0, accuracy=0.0
            )
            self.log_db.add_log("training_started", f"Training started at {self.output_dir}", checkpoint_id)
            self.logger.info("training_args : %s", training_args)
            # Instantiate Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_ds,
                eval_dataset=self.eval_ds,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False, pad_to_multiple_of=8),
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Start training with optional resume
            resume_arg = str(last_checkpoint) if last_checkpoint else None
            self.logger.info("Starting training loop with resume: %s", resume_arg)

            if resume_arg:
                # Remove state so we restart training from step 0 but keep weights
                for fname in ["optimizer.pt", "scheduler.pt", "trainer_state.json"]:
                    fpath = Path(resume_arg) / fname
                    if fpath.exists():
                        fpath.unlink()
                        self.logger.info(f"Removed {fname} to reset training state.")

                # Pass None so Trainer won't expect the deleted files
                resume_arg = None

            trainer.train(resume_from_checkpoint=resume_arg)
            trainer.save_model(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            # Success logging
            self.logger.info("Training completed successfully.")
            self.checkpoint_db.update_checkpoint(checkpoint_id, epoch=num_train_epochs, train_loss=0.0, val_loss=0.0, accuracy=0.0)
            self.log_db.add_log("training_completed", "Training completed successfully.", checkpoint_id)

        except Exception as e:
            self.logger.error("Training failed: %s", e)
            if checkpoint_id is not None:
                self.log_db.add_log("training_failed", f"Training failed with error: {e}", checkpoint_id)
            raise
