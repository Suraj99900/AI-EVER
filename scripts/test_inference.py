import argparse
import os
import math
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': tok.eos_token})
    tok.pad_token = tok.eos_token
    return tok

def load_model(model_dir: str):
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

def compute_perplexity(model, tokenizer, eval_file: str, device):
    ds = load_dataset("json", data_files={"eval": eval_file}, split="eval")
    losses = []
    for ex in ds:
        text = ex.get("text") or ex.get("prompt")
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=1024).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss.item()
        losses.append(loss)
    avg = sum(losses) / len(losses) if losses else float('inf')
    return math.exp(avg)

def generate_response(model, tokenizer, prompt: str, max_new_tokens=512, temperature=0.7, top_p=0.95, repetition_penalty=1.2,
                      no_repeat_ngram_size=3, do_sample=True, num_beams=1, stop_token=None):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
    max_len = inputs.input_ids.shape[1] + max_new_tokens
    gen_kwargs = dict(
        max_length=max_len,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    else:
        gen_kwargs['num_beams'] = num_beams

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    parts = decoded.split("### Response:", 1)
    resp = parts[1].strip() if len(parts) > 1 else decoded.strip()
    if stop_token and stop_token in resp:
        resp = resp.split(stop_token)[0] + stop_token
    return resp

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def evaluate_accuracy(model, tokenizer, ref_file: str, device, gen_opts):
    total, correct = 0, 0
    mismatches = []

    with open(ref_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                full = obj.get("text", "")
                if "### Response:" not in full:
                    continue

                prompt_part, target_part = full.split("### Response:", 1)
                prompt = normalize(prompt_part + "### Response:")
                target = normalize(target_part)

                gen = normalize(generate_response(model, tokenizer, prompt, **gen_opts))

                total += 1
                if gen == target:
                    correct += 1
                else:
                    if len(mismatches) < 20:
                        mismatches.append({"prompt": prompt, "target": target, "gen": gen})
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {idx + 1} due to JSON decode error: {e}")

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy, mismatches

def main():
    p = argparse.ArgumentParser(description="Inference & Evaluation Script")
    p.add_argument('--model_dir', type=str, help="path to checkpoints")
    p.add_argument('--prompt', type=str, help="Single prompt for inference")
    p.add_argument('--eval_file', type=str, help="JSONL for perplexity evaluation")
    p.add_argument('--eval_ref', type=str, help="JSONL with {'text': full_prompt} entries")
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--stop_token', type=str, default=None)
    p.add_argument('--no_sample', action='store_true')
    p.add_argument('--beams', type=int, default=5)
    args = p.parse_args()

    if not args.model_dir:
        raise ValueError("--model_dir must be provided")

    print(f"Loading model from: {args.model_dir}")
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_opts = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=not args.no_sample,
        num_beams=args.beams,
        stop_token=args.stop_token,
    )

    if args.eval_file:
        ppl = compute_perplexity(model, tokenizer, args.eval_file, device)
        print(f"▶️  Perplexity: {ppl:.2f}")

    if args.eval_ref:
        acc, bad = evaluate_accuracy(model, tokenizer, args.eval_ref, device, gen_opts)
        print(f"▶️  Exact-match accuracy: {acc:.2f}%")
        if bad:
            print("\nMismatches (up to 5):")
            for i, m in enumerate(bad[:5], 1):
                print(f"\n[{i}] TARGET:\n{m['target']}\n→ GENERATED:\n{m['gen']}")

    if args.prompt:
        resp = generate_response(model, tokenizer, args.prompt, **gen_opts)
        print(f"\nModel Response:\n{resp}")

if __name__ == "__main__":
    main()