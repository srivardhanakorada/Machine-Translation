# merge_and_smoketest.py
import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForSeq2SeqLM

def merge_adapters(adapter_dir: str, merged_dir: str, dtype: str = "auto"):
    t0 = time.perf_counter()
    print(f"[merge] loading adapters from: {adapter_dir}")
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
        adapter_dir,
        torch_dtype=(torch.float16 if dtype == "fp16" else "auto")
    )
    print("[merge] merging LoRA into base...")
    model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    print(f"[merge] saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)

    # tokenizer: load from adapter_dir (has your special tokens/config) and save alongside merged model
    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    tok.save_pretrained(merged_dir)

    print(f"[merge] done in {time.perf_counter() - t0:.2f}s")

def smoke_test(merged_dir: str, pref: str, text: str, max_new_tokens: int, num_beams: int):
    t0 = time.perf_counter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True  # harmless speed win on Ampere+

    print(f"[test] loading merged model from: {merged_dir}")
    tok = AutoTokenizer.from_pretrained(merged_dir, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(merged_dir).to(dev)
    mdl.eval()

    prompt = pref + text
    enc = tok([prompt], return_tensors="pt")
    # send to the model's first param device (works if using model parallel later)
    infer_device = next(mdl.parameters()).device
    enc = {k: v.to(infer_device) for k, v in enc.items()}

    print(f"[test] generating… (beams={num_beams}, max_new_tokens={max_new_tokens})")
    with torch.inference_mode():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    dt = time.perf_counter() - t0
    print(f"[test] output: {decoded}")
    print(f"[test] done in {dt:.2f}s")

def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into base T5 and run a quick smoke test.")
    ap.add_argument("--adapter_dir", default="artifacts/lora_en_fr", help="Directory with saved PEFT adapters")
    ap.add_argument("--merged_dir", default="artifacts/merged_t5_fr", help="Output directory for merged model")
    ap.add_argument("--dtype", choices=["auto", "fp16"], default="auto", help="Torch dtype for merge load")
    ap.add_argument("--skip-merge", action="store_true", help="Skip merging and only smoke-test merged_dir")
    ap.add_argument("--prefix", default="translate English to French: ", help="Prefix used during training")
    ap.add_argument("--sample", default="Hello, how are you?", help="Sample text to translate")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=1)
    args = ap.parse_args()

    if not args.skip_merge:
        if not os.path.isdir(args.adapter_dir):
            raise FileNotFoundError(f"adapter_dir not found: {args.adapter_dir}")
        merge_adapters(args.adapter_dir, args.merged_dir, dtype=args.dtype)

    if not os.path.isdir(args.merged_dir):
        raise FileNotFoundError(f"merged_dir not found: {args.merged_dir}")

    smoke_test(
        merged_dir=args.merged_dir,
        pref=args.prefix,
        text=args.sample,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

if __name__ == "__main__":
    main()
