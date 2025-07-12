from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling , BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import infer_auto_device_map, dispatch_model
import torch
import os

# Config
MODEL_PATH = "../model/starcoderbase-1b"
DATA_PATH = "../data/processed/train_sql_data.jsonl"
CHECK_POINT = "../model/checkpoints"
CACHE_DIR = "../model/model_cache"



# 2. QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

# Load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# 4. Load model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True,
    return_dict=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_proj", "q_proj", "v_proj"],  # common for StarCoder
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

# Optional: Fine-tuned memory placement for limited GPU
device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "3.7GiB",    # ✅ integer key for GPU
        "cpu": "20GiB"
    }
)
model = dispatch_model(model, device_map=device_map)



def tokenize_fn(ex): 
    result = tokenizer(ex['text'], truncation=True, padding='max_length', max_length=1024)
    result["labels"] = result["input_ids"].copy()  # 🔁 Add labels
    return result

dataset = load_dataset('json', data_files=DATA_PATH)['train'].map(tokenize_fn)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model.enable_input_require_grads()  



# Training args
args = TrainingArguments(
    output_dir=CHECK_POINT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    max_steps=50,  # use more for full training
    save_steps=50,
    logging_steps=10,
    save_total_limit =1,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

checkpoint_path = os.path.join(CHECK_POINT, "trainer_state.json")
resume = checkpoint_path if os.path.exists(checkpoint_path) else None

# Resume from checkpoint if available
trainer.train(resume_from_checkpoint=resume)

# Save adapter weights
trainer.save_model(CHECK_POINT)

tokenizer.save_pretrained(CHECK_POINT)
