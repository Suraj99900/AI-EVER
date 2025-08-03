import os
import gc
import math
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class ModelInference:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or self._locate_latest_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from: {self.model_dir}")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(self.model_dir)

        # Load model
        self.model = self._load_model(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    # ---------------- Memory Helpers ---------------- #
    def _free_memory(self):
        """Free GPU & CPU memory."""
        print("[Memory] Freeing memory...")
        torch.cuda.empty_cache()
        gc.collect()
        print("[Memory] Memory freed.")

    def get_free_vram(self):
        """Get free VRAM in MB."""
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            return free_mem // (1024 * 1024)
        return 0

    # ---------------- Dynamic Token Handling ---------------- #
    def dynamic_max_new_tokens(self, prompt, model_max=2048, min_output=128, max_output=2024):
        """Adjust max_new_tokens dynamically based on prompt size and VRAM."""
        prompt_tokens = len(self.tokenizer.encode(prompt))
        print(f"[Dynamic Tokens] Prompt length: {prompt_tokens} tokens")
        available_tokens = model_max - prompt_tokens

        # Ensure min and max limits
        available_tokens = max(min_output, min(available_tokens, max_output))

        # Adjust if VRAM is low
        free_vram = self.get_free_vram()
        if free_vram < 500:  # Very low VRAM
            available_tokens = min(128, available_tokens)
        elif free_vram < 1000:  # Low VRAM
            available_tokens = min(256, available_tokens)

        print(f"[Dynamic Tokens] Prompt Tokens: {prompt_tokens}, Output Tokens: {available_tokens}")
        return available_tokens

    # ---------------- Model Loading ---------------- #
    def _locate_latest_model(self):
        base = Path(__file__).parent.parent / 'LLMModels' / 'checkpoints'
        subs = [d for d in base.iterdir() if d.is_dir()]
        latest = max(subs, key=lambda d: d.stat().st_mtime)
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
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        return PeftModel.from_pretrained(base_model, model_dir, device_map="auto")

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

    # ---------------- Inference ---------------- #
    def generate_response(self, prompt,max_new_tokens, temperature=0.1, top_p=0.95,
                          repetition_penalty=1.2, no_repeat_ngram_size=3,
                          do_sample=False, num_beams=4, stop_token=None):

        try:
            # Free previous memory
            self._free_memory()

            # Dynamically decide tokens
            print(f"[Inference] Received prompt: {prompt}")
            max_new_tokens = self.dynamic_max_new_tokens(prompt)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True,
                max_length=2048 - max_new_tokens
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
