#!/usr/bin/env python3
import argparse
import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset


def load_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir: str):
    """Load a 4-bit Quantized + LoRA‑wrapped model on GPU/CPU."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # base + adapter
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
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding="longest", max_length=1024).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss.item()
        losses.append(loss)
    avg = sum(losses) / len(losses)
    return math.exp(avg)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = True,
    num_beams: int = None,
    stop_token: str = None,
):
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)

    max_len = inputs.input_ids.shape[1] + max_new_tokens
    gen_kwargs = dict(
        max_length=max_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if not do_sample and num_beams:
        gen_kwargs.pop('temperature')
        gen_kwargs.pop('top_p')
        gen_kwargs['num_beams'] = num_beams

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # only split once
    parts = decoded.split("### Response:", 1)
    resp = parts[1].strip() if len(parts) > 1 else decoded
    if stop_token and stop_token in resp:
        resp = resp.split(stop_token)[0] + stop_token
    return resp


def main():
    p = argparse.ArgumentParser(description="DeepSeek‑Coder Inference/Eval")
    p.add_argument('--model_dir',    type=str, help="path to checkpoints")
    p.add_argument('--prompt',       type=str, help="full prompt with markers")
    p.add_argument('--eval_file',    type=str, help="JSONL for perplexity")
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--temperature',  type=float, default=0.7)
    p.add_argument('--top_p',        type=float, default=0.95)
    p.add_argument('--stop_token',   type=str, default=None)
    p.add_argument('--no_sample',    action='store_true', help="use beam search")
    p.add_argument('--beams',        type=int, default=5, help="# beams if no_sample")
    args = p.parse_args()

    # locate latest if not given
    base = os.path.join(os.path.dirname(__file__), '..', 'LLMModels', 'checkpoints')
    if args.model_dir:
        model_dir = args.model_dir
    else:
        subs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        model_dir = max(subs, key=os.path.getmtime)

    print("Loading tokenizer…")
    tokenizer = load_tokenizer(model_dir)

    print("Loading model…")
    model = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.eval_file:
        ppl = compute_perplexity(model, tokenizer, args.eval_file, device)
        print(f"Perplexity: {ppl:.2f}")
        return

    if not args.prompt:
        raise ValueError("Please pass --prompt for inference")

    resp = generate_response(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        num_beams=args.beams,
        stop_token=args.stop_token
    )
    print(resp)


if __name__ == "__main__":
    main()
