"""Merge a LoRA adapter into its base model and save."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", required=True, help="LoRA adapter path")
    p.add_argument("--out", required=True, help="Output directory for merged model")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda", help="Device for merge (cuda or cpu)")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)

    print("Saving tokenizer (from adapter dir to preserve any tokenizer changes)...")
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print("Done.")


if __name__ == "__main__":
    main()
