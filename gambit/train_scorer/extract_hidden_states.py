"""Extract step-boundary hidden states for sequence-scorer training (Appendix A.1).

Given a JSONL of sampled reasoning traces, this script produces the per-trace
step sequences (h_1, ..., h_T) that `train_sequence_scorer.py` consumes.

For each trace:
  1. Rebuild the exact text the model saw: chat-templated prompt + response.
  2. Locate reasoning-step boundaries -- occurrences of "\\n\\n" inside the
     <think> block.
  3. Run a single forward pass and take the last-layer hidden state at each
     boundary token. Boundaries are mapped to tokens via the fast tokenizer's
     offset mapping, so no re-decoding is needed.
  4. Label the trace with binary correctness of its final extracted answer.

Input JSONL, one object per line:
    {"question": str,
     "response": str,          # full generation, containing the <think> block
     "ground_truth": str,
     "extracted_answer": str}  # optional; re-extracted from `response` if absent

Output: sharded .pt files (`shard_00000.pt`, ...), each a list of dicts:
    {"select_hidden_states": Tensor[T, D] float16,   # one row per step boundary
     "num_steps": int,
     "question": str,
     "ground_truth": str,
     "extracted_answer": str,
     "is_correct": bool}

Usage:
    python gambit/train_scorer/extract_hidden_states.py \\
      --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \\
      --input_path traces.jsonl \\
      --output_dir hidden_states/train \\
      --shard_size 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from evaluator import extract_answer, math_equal  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract step-boundary hidden states for the sequence scorer")
    p.add_argument('--model_name', type=str, required=True,
                   help='HF model id or local path (must match the model that '
                        'generated the traces)')
    p.add_argument('--input_path', type=str, required=True,
                   help='Input JSONL of traces')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory for the output .pt shards')
    p.add_argument('--start_idx', type=int, default=0)
    p.add_argument('--end_idx', type=int, default=None)
    p.add_argument('--shard_size', type=int, default=1000,
                   help='Traces per output shard (default: 1000)')
    p.add_argument('--max_tokens', type=int, default=0,
                   help='Skip traces longer than this many tokens; 0 = no limit')
    p.add_argument('--dtype', type=str, default='bfloat16',
                   choices=['bfloat16', 'float16', 'float32'])
    return p.parse_args()


def build_full_text(question: str, response: str, tokenizer) -> tuple[str, int]:
    """Return (prompt + response, char offset where the response begins)."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + response, len(prompt)


def step_boundary_positions(full_text: str, prompt_len: int,
                            offsets: list[tuple[int, int]]) -> list[int]:
    """Token indices of the step boundaries inside the <think> block.

    Mirrors the engine, which captures a hidden state whenever the sampled token
    *contains* "\\n\\n" (tokenizers routinely merge it with adjacent text, e.g.
    ".\\n\\n"). Selecting the containing token — rather than the one before it —
    is what keeps training data aligned with inference-time capture.

    The thinking block is delimited by <think>/</think>. Some chat templates emit
    the opening tag as part of the prompt, so its absence in the response is not
    an error -- fall back to the start of the response.
    """
    open_tag = full_text.find("<think>")
    think_start = open_tag + len("<think>") if open_tag != -1 else prompt_len
    close_tag = full_text.find("</think>", think_start)
    think_end = close_tag if close_tag != -1 else len(full_text)

    return [
        i for i, (s, e) in enumerate(offsets)
        if think_start <= s < think_end and "\n\n" in full_text[s:e]
    ]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{args.model_name} has no fast tokenizer; offset mapping is required.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    model.eval()

    with open(args.input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = records[args.start_idx:args.end_idx]
    print(f"Loaded {len(records)} traces from {args.input_path}")

    shard, shard_idx, n_written, n_skipped = [], 0, 0, 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(shard, path)
        print(f"  wrote {path} ({len(shard)} traces)")
        shard, shard_idx = [], shard_idx + 1

    for rec in tqdm(records, desc="Extracting"):
        question = rec["question"]
        response = rec["response"]
        ground_truth = str(rec.get("ground_truth", rec.get("answer", "")))
        predicted = rec.get("extracted_answer") or extract_answer(response, "math")

        full_text, prompt_len = build_full_text(question, response, tokenizer)
        enc = tokenizer(full_text, return_offsets_mapping=True,
                        add_special_tokens=False)
        offsets = enc["offset_mapping"]
        input_ids = torch.tensor([enc["input_ids"]], device=model.device)

        if args.max_tokens and input_ids.shape[1] > args.max_tokens:
            n_skipped += 1
            continue

        positions = step_boundary_positions(full_text, prompt_len, offsets)
        if not positions:
            n_skipped += 1
            continue

        with torch.no_grad():
            hidden = model.model(input_ids).last_hidden_state[0]   # [T_tok, D]
        selected = hidden[positions].to(torch.float16).cpu()       # [T_steps, D]
        del hidden
        torch.cuda.empty_cache()

        shard.append({
            "select_hidden_states": selected,
            "num_steps": selected.shape[0],
            "question": question,
            "ground_truth": ground_truth,
            "extracted_answer": predicted,
            "is_correct": bool(predicted) and math_equal(predicted, ground_truth),
        })
        n_written += 1
        if len(shard) >= args.shard_size:
            flush()

    flush()
    print(f"Done: {n_written} traces written, {n_skipped} skipped -> {args.output_dir}")


if __name__ == "__main__":
    main()
