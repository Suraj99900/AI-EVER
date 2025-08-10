import os
import gc
import math
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig ,TextIteratorStreamer
from peft import PeftModel
import logging
import threading



class ModelInference:
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "model_inference.log"

    def __init__(self, model_dir=None):
        # Setting the log
        logging.basicConfig(
            filename=self.log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(console)

        self.model_dir = model_dir or self._locate_latest_model()
        logging.info(f"Loading model from: {self.model_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(self.model_dir)

        # Load model
        self.model = self._load_model(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    # ---------------- Memory Helpers ---------------- #
    def _free_memory(self):
        """Free GPU & CPU memory."""
        logging.info("[Memory] Freeing memory...")
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("[Memory] Memory freed.")

    def get_free_vram(self):
        """Get free VRAM in MB."""
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            return free_mem // (1024 * 1024)
        return 0

    # ---------------- Dynamic Token Handling ---------------- #
    def dynamic_max_new_tokens(self, prompt, model_max=2048, min_output=128, max_output=1500):
        """
        Adjust max_new_tokens dynamically based on prompt size, VRAM, and content type.
        Gives extra tokens for code-heavy tasks.
        """
        prompt_tokens = len(self.tokenizer.encode(prompt))
        logging.info(f"[Dynamic Tokens] Prompt length: {prompt_tokens} tokens")

        # Detect if the task is code-heavy
        code_keywords = ["html", "javascript", "css", "python", "code", "script", "function", "api","sql"]
        is_code_task = any(kw in prompt.lower() for kw in code_keywords)

        # Base available tokens
        available_tokens = model_max - prompt_tokens

        # Increase allowance for code tasks
        if is_code_task:
            available_tokens += 500  # give a bonus budget for code generation

        # Ensure within min/max limits
        available_tokens = max(min_output, min(available_tokens, max_output))

        # Adjust if VRAM is low
        free_vram = self.get_free_vram()
        if free_vram < 500:
            available_tokens = min(256, available_tokens)
        elif free_vram < 1000:
            available_tokens = min(512, available_tokens)

        logging.info(f"[Dynamic Tokens] Task: {'Code' if is_code_task else 'General'}, "
                    f"Prompt Tokens: {prompt_tokens}, Output Tokens: {available_tokens}")

        return available_tokens


    # ---------------- Model Loading ---------------- #
    def _locate_latest_model(self):
        # Locate the latest model directory if not then pass base directory
        base = Path(__file__).parent.parent / 'LLMModels' / 'checkpoints'
        logging.info(f"Path Of the Base Model :- {base}")

        subs = [d for d in base.iterdir() if d.is_dir()]
        logging.info(f"Subdirectories found: {subs}")
        if not subs:
            logging.info("No model directories found. Using base directory.")
            MODEL_DIR  = Path(__file__).parent.parent / "LLMModels"/"deepseek-coder-1.3b-base"
            return MODEL_DIR
        latest = max(subs, key=lambda d: d.stat().st_mtime)
        logging.info(f"Latest model directory: {latest}")
        return str(latest)

    def _load_tokenizer(self, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_model(self, model_dir):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Check if it's a PEFT adapter (has adapter_config.json)
        adapter_config_path = Path(model_dir) / "adapter_config.json"

        if adapter_config_path.exists():
            logging.info("[Model Loader] Detected LoRA adapter. Loading base + adapter...")
            # You must define where your base model is stored
            base_model_path = "/mnt/New Volume/AI-EVER/LLMModels/deepseek-coder-1.3b-base"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                quantization_config=bnb, 
                device_map="auto", 
                trust_remote_code=True
            )
            return PeftModel.from_pretrained(base_model, model_dir, device_map="auto")

        else:
            logging.info("[Model Loader] Detected full model. Loading directly...")
            return AutoModelForCausalLM.from_pretrained(
                model_dir, 
                quantization_config=bnb, 
                device_map="auto", 
                trust_remote_code=True
            )
    # ---------------- Perplexity ---------------- #
    def compute_perplexity(self, eval_file):
        ds = load_dataset("json", data_files={"eval": eval_file}, split="eval")
        losses = []
        for ex in ds:
            text = ex.get("text") or ex.get("prompt")
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding="longest", max_length=1024
            ).to(self.device)
            with torch.no_grad():
                loss = self.model(**inputs, labels=inputs.input_ids).loss.item()
            losses.append(loss)
        avg = sum(losses) / len(losses)
        return math.exp(avg)
    

    # --------------- Inferance like stream of text genration ----------------
    def generate_response_stream(self, prompt,max_new_tokens, temperature=0.1, top_p=0.95,
                          repetition_penalty=1.2, no_repeat_ngram_size=3,
                          do_sample=False, num_beams=4, stop_token=None):
        self._free_memory()
        logging.info(f"[Streaming] Prompt received: {prompt}")

        max_new_tokens = self.dynamic_max_new_tokens(prompt)

        # Tokenize once
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        logging.info(f"Final Token used : {max_new_tokens} ")
        # Generation arguments
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            streamer=streamer
        )

        # Run generation in background
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they arrive
        for token in streamer:
            yield token

    # ---------------- Inference ---------------- #
    def generate_response(self, prompt,max_new_tokens, temperature=0.1, top_p=0.95,
                          repetition_penalty=1.2, no_repeat_ngram_size=3,
                          do_sample=False, num_beams=4, stop_token=None):

        try:
            # Free previous memory
            self._free_memory()

            # Dynamically decide tokens
            logging.info(f"[Inference] Received prompt: {prompt}")
            max_new_tokens = self.dynamic_max_new_tokens(prompt)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True,
                max_length=max_new_tokens
            ).to(self.device)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
                
            ).to(self.device)

            # Generation settings
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if not do_sample and num_beams:
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs["num_beams"] = num_beams

            # Generate
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = self.model.generate(**inputs, **gen_kwargs)

            # Decode
            decoded = self.tokenizer.decode(
                output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Remove prompt part
            decoded = decoded[len(prompt):]

            # Apply stop token
            if stop_token and stop_token in decoded:
                decoded = decoded.split(stop_token)[0] + stop_token

            return decoded

        except torch.cuda.OutOfMemoryError:
            print("[Error] CUDA OOM - Retrying with fewer tokens...")
            self._free_memory()
            return self.generate_response(prompt, temperature, top_p,
                                          repetition_penalty, no_repeat_ngram_size,
                                          do_sample, num_beams, stop_token)

        finally:
            # Always free memory after inference
            self._free_memory()
