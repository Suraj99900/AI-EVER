# models/ModelTraining.py
import os
import math
import torch
import datetime
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sql.CheckpointTrackMaster import CheckpointTrackMaster
from sql.AIEverLog import AIEverLog

class ModelTrainer:
    def __init__(self, model_dir = "../LLMModels/deepseek-coder-1.3b-base", data_path = "../data/processed/data/processed/train_data.jsonl", base_output_path="../LLMModels/checkpoints/" , sql_db_path="../data/processed/train_sql_data.jsonl",is_sql=False):
        self.model_dir = model_dir
        self.data_path = is_sql if data_path else sql_db_path
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_output_path, f"deepseek-1.3b-lora-{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.checkpoint_db = CheckpointTrackMaster()
        self.log_db = AIEverLog()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def load_model_and_tokenizer(self):
        self.logger.info("Loading model and tokenizer...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
        base_model = prepare_model_for_kbit_training(base_model)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none", task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(base_model, lora_cfg)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model.to(self.device)

    def prepare_data(self):
        self.logger.info("Preparing dataset...")
        dataset = load_dataset("json", data_files=self.data_path, split="train")
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = splits["train"], splits["test"]

        def tokenize_fn(batch):
            toks = self.tokenizer(batch["text"], truncation=True, padding="longest", max_length=512)
            input_ids = toks["input_ids"]
            labels = [[tok if tok != self.tokenizer.pad_token_id else -100 for tok in seq] for seq in input_ids]
            return {"input_ids": input_ids, "attention_mask": toks["attention_mask"], "labels": labels}

        self.train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        self.eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def compute_metrics(self, eval_preds,max_steps=20,per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-4,num_train_epochs=10,logging_steps=2,eval_steps=5,warmup_steps=5,save_steps=20,save_strategy="steps",save_total_limit=1,metric_for_best_model="eval_loss",fp16=torch.cuda.is_available(),bf16=False,greater_is_better=False,optim="adamw_torch",report_to=[],logging_dir=None):
        logits, labels = eval_preds
        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        return {"perplexity": math.exp(loss.item())}

    def train(self):
        self.logger.info("Initializing training configuration...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size= self.per_device_train_batch_size,
            gradient_accumulation_steps= self.gradient_accumulation_steps,
            max_steps= self.max_steps,
            learning_rate= self.learning_rate,
            num_train_epochs= self.num_train_epochs,
            logging_steps= self.logging_steps,
            eval_steps= self.eval_steps,
            warmup_steps= self.warmup_steps,
            save_steps= self.save_steps,
            save_strategy= self.save_strategy,
            save_total_limit= self.save_total_limit,
            metric_for_best_model= self.metric_for_best_model,
            fp16= self.fp16,
            bf16=  self.bf16,
            greater_is_better= self.greater_is_better,
            optim= self.optim,
            report_to= self.report_to,
            logging_dir= self.logging_dir if self.logging_dir else os.path.join(self.output_dir, "logs")
        )

        last_checkpoint = None
        if os.path.isdir(self.output_dir):
            checkpoints = [os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir)
                           if d.startswith("checkpoint")]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getctime)
                self.logger.info(f"Resuming training from: {last_checkpoint}")

        checkpoint_id = self.checkpoint_db.add_checkpoint(
            model_name=os.path.basename(self.model_dir),
            checkpoint_dir=self.output_dir,
            epoch=0, train_loss=0.0, val_loss=0.0, accuracy=0.0
        )

        self.log_db.add_log("training_started", f"Training started at {self.output_dir}", checkpoint_id)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False, pad_to_multiple_of=8),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        self.logger.info("Training completed. Saving log...")
        self.checkpoint_db.update_checkpoint(checkpoint_id, epoch=10, train_loss=0.0, val_loss=0.0, accuracy=0.0)
        self.log_db.add_log("training_completed", "Training completed successfully.", checkpoint_id)

