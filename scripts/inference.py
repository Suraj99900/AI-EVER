#!/usr/bin/env python3
import argparse
import os
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import load_dataset

def compute_perplexity(model, tokenizer, eval_path, device):
    ds = load_dataset("json", data_files={"eval": eval_path}, split="eval")
    losses = []
    for ex in ds:
        text = ex.get("text") or ex.get("prompt")
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="longest", max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            losses.append(outputs.loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference or evaluation on fine-tuned DeepSeek-Coder model"
    )
    parser.add_argument(
        '--model_dir', type=str, default=None,
        help='Path to your PEFT-fine-tuned model folder'
    )
    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Prompt text including ### Instruction: and ### Response:'
    )
    parser.add_argument(
        '--language', type=str, default=None,
        help='Optional language tag (e.g., python, sql, javascript)'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=512,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_p', type=float, default=0.95,
        help='Nucleus sampling top-p'
    )
    parser.add_argument(
        '--stop_token', type=str, default=None,
        help='Optional stop token to truncate output'
    )
    parser.add_argument(
        '--eval_file', type=str, default=None,
        help='Optional JSONL file for perplexity evaluation'
    )
    args = parser.parse_args()

    # Determine model directory

    def get_latest_checkpoint(base_dir):
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Checkpoint base directory not found: {base_dir}")

        subdirs = [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        if not subdirs:
            raise FileNotFoundError("No subdirectories found in checkpoint base directory")

        latest = max(subdirs, key=os.path.getmtime)
        return latest
    
    checkpoint_base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints')
    )
    model_dir = args.model_dir or get_latest_checkpoint(checkpoint_base_dir)
    print(f"Using model checkpoint: {model_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token

    # BitsAndBytes config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load base model and wrap with LoRA adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, model_dir, device_map='auto')
    model.to(device).eval()

    # Evaluation mode
    if args.eval_file:
        ppl = compute_perplexity(model, tokenizer, args.eval_file, device)
        print(f"Perplexity: {ppl:.2f}")
        return

    # Inference mode
    if not args.prompt:
        raise ValueError("--prompt is required for inference mode")

    # Ensure prompt markers present
    prompt = args.prompt.strip()
    if "### Response:" not in prompt:
        raise ValueError("Prompt must include '### Response:' marker.")

    # Tokenize input without extra code fences
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)

    # Generation with anti-repetition
    max_length = inputs.input_ids.shape[1] + args.max_new_tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and extract response
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = raw.split("### Response:")[-1].strip()

    # Stop-token truncation
    if args.stop_token:
        idx = text.find(args.stop_token)
        if idx != -1:
            text = text[: idx + len(args.stop_token)]

    # Wrap output if needed
    if args.language:
        print(f"```{args.language}\n{text}\n```")
    else:
        print(text)

if __name__ == "__main__":
    main()
