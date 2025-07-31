import os
import math
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class ModelInference:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or self._locate_latest_model()
        self.tokenizer = self._load_tokenizer(self.model_dir)
        self.model = self._load_model(self.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _locate_latest_model(self):
        base = Path(__file__).parent.parent / 'LLMModels' / 'checkpoints'
        subs = [d for d in base.iterdir() if d.is_dir()]
        latest = max(subs, key=lambda d: d.stat().st_mtime)
        return str(latest)

    def _load_tokenizer(self, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, model_dir):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb, device_map='auto', trust_remote_code=False
        )
        model = PeftModel.from_pretrained(base, model_dir, device_map='auto')
        model.eval()
        return model

    def compute_perplexity(self, eval_file):
        ds = load_dataset("json", data_files={"eval": eval_file}, split="eval")
        losses = []
        for ex in ds:
            text = ex.get("text") or ex.get("prompt")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=1024).to(self.device)
            with torch.no_grad():
                loss = self.model(**inputs, labels=inputs.input_ids).loss.item()
            losses.append(loss)
        avg = sum(losses) / len(losses)
        return math.exp(avg)

    def generate_response(self, prompt, max_new_tokens=512, temperature=0.2, top_p=0.95,
                          repetition_penalty=1.2, no_repeat_ngram_size=3, do_sample=False,
                          num_beams=None, stop_token=None):
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        max_len = inputs.input_ids.shape[1] + max_new_tokens
        gen_kwargs = dict(
            max_length=max_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if not do_sample and num_beams:
            gen_kwargs.pop('temperature')
            gen_kwargs.pop('top_p')
            gen_kwargs['num_beams'] = num_beams

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        parts = decoded.split("### Response:", 1)
        resp = parts[1].strip() if len(parts) > 1 else decoded
        if stop_token and stop_token in resp:
            resp = resp.split(stop_token)[0] + stop_token
        return resp
