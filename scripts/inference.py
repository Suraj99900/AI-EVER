# scripts/inference.py
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned StarCoder model")
    parser.add_argument(
        '--prompt', type=str, required=True,
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
    args = parser.parse_args()

    # Locate the checkpoint directory
    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints')
    )

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model.eval()

    # Build full prompt with optional language tag
    full_prompt = ''
    if args.language:
        full_prompt += f"[LANG:{args.language}]\n"
    full_prompt += args.prompt

    # Tokenize input
    inputs = tokenizer(
        full_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode text
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the Response section
    if "### Response:" in raw:
        text = raw.split("### Response:")[-1].strip()
    else:
        text = raw

    # Optional post‑processing: truncate at stop_token
    if args.stop_token:
        idx = text.find(args.stop_token)
        if idx != -1:
            text = text[: idx + len(args.stop_token)]

    # Wrap in code fences if language specified
    if args.language:
        print(f"```{args.language}\n{text}\n```")
    else:
        print(text)
