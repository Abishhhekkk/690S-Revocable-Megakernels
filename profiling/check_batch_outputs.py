#!/usr/bin/env python3
"""Check that MPK's batched decode produces correct tokens for every request.

Usage:
    python profiling/check_batch_outputs.py ~/mirage/tests/ci-tests/run_batch_perf.py \
        --model Qwen/Qwen3-1.7B --max-num-batched-requests 8 \
        --max-num-batched-tokens 8 --max-seq-length 256 --ignore-eos --output-dir build

run_batch_perf.py gives every request the same 1-token prompt and decodes
greedily, so all requests must produce identical tokens. Only request 0 sat
on KV page 0, where the sm80 PAGE_STRIDE_ROWS bug was harmless; requests 1+
are the ones the fix affects. We also compare request 0 against Hugging Face
greedy generation (bf16 rounding can make a long run diverge, so we report
the length of the matching prefix).

Exit code 0 if every request matches request 0, else 1.
"""
import os
import runpy
import sys

import torch

captured = {}


def grab_main_locals(frame, event, _arg):
    if event == "return" and frame.f_code.co_name == "main" and "tokens" in frame.f_locals:
        captured.update(frame.f_locals)


def prefix_match(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + sys.argv[2:]

    sys.setprofile(grab_main_locals)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.setprofile(None)
    if "tokens" not in captured:
        sys.exit("[check] could not capture tokens from main()")

    tokens, step = captured["tokens"].cpu(), captured["step"].cpu()
    tokenizer, args = captured["tokenizer"], captured["args"]
    R = tokens.shape[0]
    lens = [int(step[r]) + 1 for r in range(R)]
    rows = [tokens[r, :lens[r]].tolist() for r in range(R)]
    print(f"\n[check] {R} requests, sequence lengths {lens}")

    ok = True
    for r in range(R):
        m = prefix_match(rows[0], rows[r])
        same = m == len(rows[0]) == len(rows[r])
        ok &= same
        print(f"[check] request {r}: {'MATCHES' if same else 'DIFFERS'} request 0 "
              f"(first {m}/{len(rows[r])} tokens equal)")
        if not same:
            print(f"        text: {tokenizer.decode(rows[r][:48])!r}")

    # Hugging Face greedy reference for request 0.
    del captured["model"]
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    prompt = torch.tensor([rows[0][:1]], device="cuda")
    with torch.no_grad():
        ref = hf.generate(prompt, max_new_tokens=len(rows[0]) - 1, do_sample=False,
                          min_new_tokens=len(rows[0]) - 1)[0].tolist()
    m = prefix_match(rows[0], ref)
    print(f"[check] request 0 vs Hugging Face greedy: first {m}/{len(rows[0])} tokens equal")
    print(f"        MPK: {tokenizer.decode(rows[0][:48])!r}")
    print(f"        HF : {tokenizer.decode(ref[:48])!r}")

    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(all {R} requests identical: {ok}; HF prefix match {m} tokens)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
