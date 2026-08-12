#!/usr/bin/env python3
"""
Compute pass@k from the question JSON files written by `benchmark_eval.py`.

This script is memory-safe for very large question JSON files: it does NOT
json-load the whole file. Instead it parses a small header window and streams
the file to find `"is_trace_correct": true/false`.

Pass@k definition (sampling-based):
  - For each question, form a pool of answers by concatenating traces across
    one or more run directories (e.g., 2 runs x 256 traces = 512 pool size).
  - For each k, repeatedly (default 100 trials) sample k traces WITHOUT
    replacement from that pool and mark the trial success if any sampled trace
    is correct. pass@k is the mean success rate over trials.

Examples:
  python gambit/tests/compute_pass_at_n.py --run-dir RUN_A --k 256
  python gambit/tests/compute_pass_at_n.py --run-dir RUN_A --run-dir RUN_B --k 1,2,4,8,16,32,64,128,256

Pass --allow-missing-questions when run directories cover different subsets of
questions, and --suppress-parse-errors to skip unreadable files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import random
import json
from dataclasses import dataclass
from glob import glob
from typing import Optional


@dataclass(frozen=True)
class QuestionPool:
    question_index: int
    question: Optional[str]
    correct_flags: list[bool]  # pooled across run dirs
    token_lengths: list[int]  # pooled across run dirs (aligned with correct_flags)
    extracted_answers: list[Optional[str]]  # pooled across run dirs (aligned)
    ground_truth: Optional[str]


_HEADER_FIELD_PATTERNS = {
    "question_index": re.compile(r'"question_index"\s*:\s*(\d+)\s*,?'),
    "question": re.compile(r'"question"\s*:\s*("(?:(?:\\.)|[^"\\])*")\s*,?'),
    "total_traces": re.compile(r'"total_traces"\s*:\s*(\d+)\s*,?'),
    "num_correct_traces": re.compile(r'"num_correct_traces"\s*:\s*(\d+)\s*,?'),
    "ground_truth": re.compile(r'"ground_truth"\s*:\s*("(?:(?:\\.)|[^"\\])*")\s*,?'),
}

# Parse all needed per-generation fields in a single streaming pass.
# Groups are mutually exclusive per match:
#   - group(1): true/false for is_trace_correct
#   - group(2): digits for token_length
#   - group(3): extracted_answer as `null` or a JSON string (including quotes)
_TRACE_FIELD_RE = re.compile(
    rb'(?<!\\)"(?:'
    rb'is_trace_correct"\s*:\s*(true|false)'
    rb'|token_length"\s*:\s*(\d+)'
    rb'|extracted_answer"\s*:\s*(null|"(?:\\.|[^"\\])*")'
    rb')\s*,?'
)


def _parse_header_fields(
    path: str, header_bytes: int = 256 * 1024
) -> tuple[Optional[int], Optional[str], Optional[int], Optional[int], Optional[str]]:
    with open(path, "rb") as f:
        head = f.read(header_bytes)
    text = head.decode("utf-8", errors="replace")

    qidx = None
    question = None
    total_traces = None
    num_correct = None
    ground_truth = None

    m = _HEADER_FIELD_PATTERNS["question_index"].search(text)
    if m:
        qidx = int(m.group(1))

    m = _HEADER_FIELD_PATTERNS["question"].search(text)
    if m:
        try:
            question = json.loads(m.group(1))
        except Exception:
            question = None

    m = _HEADER_FIELD_PATTERNS["total_traces"].search(text)
    if m:
        total_traces = int(m.group(1))

    m = _HEADER_FIELD_PATTERNS["num_correct_traces"].search(text)
    if m:
        num_correct = int(m.group(1))

    m = _HEADER_FIELD_PATTERNS["ground_truth"].search(text)
    if m:
        try:
            ground_truth = json.loads(m.group(1))
        except Exception:
            ground_truth = None

    return qidx, question, total_traces, num_correct, ground_truth


def _norm_question_key(q: str) -> str:
    # Keep it simple: exact text match after strip.
    return (q or "").strip()


def _norm_answer_for_compare(a: Optional[str]) -> Optional[str]:
    if a is None:
        return None
    s = str(a).strip()
    if not s:
        return None
    # Canonicalise only genuine integers (optional sign, thousands separators,
    # leading zeros), so "070" == "70" and "1,234" == "1234".
    #
    # Deliberately NOT a general strip-all-non-digits: that would collapse
    # "1/2" and "8sqrt(10)" to "12" and "810", silently equating distinct
    # symbolic answers. Anything that is not a plain integer is compared as
    # case-folded text.
    compact = s.replace(",", "").replace(" ", "")
    if re.fullmatch(r"[+-]?\d+", compact):
        return str(int(compact))
    return s.lower()


def _load_answer_key_jsonl(path: str) -> dict[str, str]:
    """
    Loads a JSONL file containing at least:
      - {"question": "...", "answer": "..."}
    or meta rows that include the same keys.
    Returns mapping from normalized question text -> raw answer string.
    """
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("question")
            a = rec.get("answer")
            if q is None or a is None:
                continue
            out[_norm_question_key(str(q))] = str(a)
    return out


def _collect_trace_fields(
    path: str,
    chunk_size: int = 1 * 1024 * 1024,
) -> tuple[list[bool], list[int], list[Optional[str]]]:
    """
    Collect per-generation fields in encounter order, streaming the file:
      - is_trace_correct (bool)
      - token_length (int)
      - extracted_answer (Optional[str])

    We do not json-load the file. We scan for the exact JSON key occurrences.
    `benchmark_eval.py` writes one `token_length`, one `is_trace_correct`, and
    one `extracted_answer` per generation entry, so the three lists align.
    """
    flags: list[bool] = []
    token_lengths: list[int] = []
    extracted_answers: list[Optional[str]] = []
    overlap = b""
    # Important: defer matches that START near the end of the buffer to avoid
    # missing boundary-crossing matches. `keep` must be >= max match length.
    # extracted_answer is typically very short; 64 KiB is a safe upper bound.
    keep = 64 * 1024

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf = overlap + chunk
            limit = max(0, len(buf) - keep)
            for m in _TRACE_FIELD_RE.finditer(buf):
                # Defer matches that start in the last `keep` bytes.
                if m.start() >= limit:
                    continue
                if m.group(1) is not None:
                    flags.append(m.group(1) == b"true")
                elif m.group(2) is not None:
                    token_lengths.append(int(m.group(2)))
                else:
                    raw = m.group(3)
                    if raw == b"null":
                        extracted_answers.append(None)
                    else:
                        try:
                            extracted_answers.append(json.loads(raw.decode("utf-8")))
                        except Exception:
                            extracted_answers.append(None)
            overlap = buf[-keep:] if len(buf) > keep else buf

    for m in _TRACE_FIELD_RE.finditer(overlap):
        if m.group(1) is not None:
            flags.append(m.group(1) == b"true")
        elif m.group(2) is not None:
            token_lengths.append(int(m.group(2)))
        else:
            raw = m.group(3)
            if raw == b"null":
                extracted_answers.append(None)
            else:
                try:
                    extracted_answers.append(json.loads(raw.decode("utf-8")))
                except Exception:
                    extracted_answers.append(None)

    return flags, token_lengths, extracted_answers


def _parse_k_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty --k")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ks: list[int] = []
    for p in parts:
        k = int(p)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ks.append(k)
    # preserve order, de-dup
    seen: set[int] = set()
    out: list[int] = []
    for k in ks:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _index_question_files(run_dir: str, pattern: str) -> dict[int, str]:
    paths = sorted(glob(os.path.join(run_dir, pattern)))
    out: dict[int, str] = {}
    for p in paths:
        qidx, _, _, _, _ = _parse_header_fields(p)
        if qidx is None:
            raise ValueError(f"Could not parse question_index from {p}")
        if qidx in out:
            raise ValueError(f"Duplicate question_index={qidx} under {run_dir}: {out[qidx]} and {p}")
        out[qidx] = p
    return out




def _build_pools_impl(
    run_dirs: list[str],
    pattern: str,
    *,
    allow_missing_questions: bool,
    suppress_parse_errors: bool,
    answer_key: Optional[dict[str, str]] = None,
) -> list[QuestionPool]:
    indexed = [_index_question_files(d, pattern) for d in run_dirs]
    if not indexed:
        return []

    all_qidx = sorted(set().union(*[set(m.keys()) for m in indexed]))
    if not all_qidx:
        return []

    if allow_missing_questions:
        qidxs = all_qidx
    else:
        # Require same questions exist in all run dirs to form consistent pools.
        common = set(indexed[0].keys())
        for m in indexed[1:]:
            common &= set(m.keys())
        qidxs = sorted(common)

    if not allow_missing_questions:
        missing_report = []
        for q in all_qidx:
            missing = [run_dirs[i] for i, m in enumerate(indexed) if q not in m]
            if missing:
                missing_report.append((q, missing))
        if missing_report:
            lines = ["Some questions are missing from one or more run dirs (cannot pool):"]
            for q, miss in missing_report[:20]:
                lines.append(f"  question_index={q} missing in {len(miss)}/{len(run_dirs)} run dirs")
                for d in miss:
                    lines.append(f"    - {d}")
            if len(missing_report) > 20:
                lines.append(f"  ... and {len(missing_report) - 20} more")
            raise ValueError("\n".join(lines))

    pools: list[QuestionPool] = []
    for qidx in qidxs:
        flags_all: list[bool] = []
        toks_all: list[int] = []
        ans_all: list[Optional[str]] = []
        gt: Optional[str] = None
        qtext: Optional[str] = None

        for i, d in enumerate(run_dirs):
            path = indexed[i].get(qidx)
            if path is None:
                continue
            try:
                qidx2, question, total_declared, correct_declared, ground_truth = _parse_header_fields(path)
                if qidx2 is not None and qidx2 != qidx:
                    raise ValueError(f"Header question_index mismatch in {path}: got {qidx2}, expected {qidx}")

                if qtext is None:
                    qtext = question
                elif question is not None and question != qtext and not suppress_parse_errors:
                    raise ValueError(f"question text mismatch for qidx={qidx} across runs")

                # Determine ground truth:
                if answer_key is not None and qtext is not None:
                    gt2 = answer_key.get(_norm_question_key(qtext))
                    if gt2 is not None:
                        gt = gt2
                    elif gt is None:
                        gt = ground_truth
                else:
                    if gt is None:
                        gt = ground_truth
                    elif (
                        not allow_missing_questions
                        and ground_truth is not None
                        and ground_truth != gt
                    ):
                        # Pooled runs must agree on the ground truth.
                        raise ValueError(
                            f"ground_truth mismatch for qidx={qidx}: {gt!r} vs {ground_truth!r} in {path}"
                        )

                flags, token_lengths, extracted_answers = _collect_trace_fields(path)
                if len(flags) != len(token_lengths) or len(flags) != len(extracted_answers):
                    raise ValueError(
                        f"Mismatched fields in {path}: is_trace_correct={len(flags)} token_length={len(token_lengths)} extracted_answer={len(extracted_answers)}"
                    )
                if total_declared is not None and total_declared != len(flags):
                    raise ValueError(f"Declared total_traces={total_declared} but observed {len(flags)} in {path}")

                # If we have an external answer key, recompute correctness from extracted_answer.
                if answer_key is not None and gt is not None:
                    gt_norm = _norm_answer_for_compare(gt)
                    flags = [
                        (_norm_answer_for_compare(a) is not None and _norm_answer_for_compare(a) == gt_norm)
                        for a in extracted_answers
                    ]
                else:
                    if correct_declared is not None and correct_declared != sum(flags):
                        raise ValueError(
                            f"Declared num_correct_traces={correct_declared} but observed {sum(flags)} in {path}"
                        )

                flags_all.extend(flags)
                toks_all.extend(token_lengths)
                ans_all.extend(extracted_answers)
            except Exception:
                if suppress_parse_errors:
                    # Skip malformed files silently in this mode.
                    continue
                raise

        if flags_all:
            pools.append(
                QuestionPool(
                    question_index=qidx,
                    question=qtext,
                    correct_flags=flags_all,
                    token_lengths=toks_all,
                    extracted_answers=ans_all,
                    ground_truth=gt,
                )
            )
    return pools


def _estimate_pass_majority_and_tokens_at_k(
    pool_correct: list[bool],
    pool_tokens: list[int],
    pool_answers: list[Optional[str]],
    *,
    ground_truth: Optional[str],
    k: int,
    trials: int,
    seed: int,
) -> tuple[float, float, float]:
    """
    Returns:
      - pass_rate: mean( any correct in sampled k ) over trials
      - majority_vote_rate: mean( majority(extracted_answer) == ground_truth ) over trials
      - avg_token_sum: mean(sum(token_length of sampled generations)) over trials

    Majority vote ignores `None` answers. If no non-None answers in the sample,
    the trial counts as incorrect. Tie-break: earliest appearance in the sampled
    order (deterministic).
    """
    if not pool_correct:
        return 0.0, 0.0, 0.0

    m = len(pool_correct)
    k_eff = min(k, m)
    rng = random.Random(seed)
    idxs = list(range(m))

    pass_successes = 0
    mv_successes = 0
    token_sum = 0

    for _ in range(trials):
        picked = rng.sample(idxs, k_eff)

        if any(pool_correct[i] for i in picked):
            pass_successes += 1
        token_sum += sum(pool_tokens[i] for i in picked)

        if ground_truth is None:
            continue

        counts: dict[str, int] = {}
        first_pos: dict[str, int] = {}
        for pos, i in enumerate(picked):
            ans = pool_answers[i]
            if ans is None:
                continue
            if ans not in counts:
                counts[ans] = 1
                first_pos[ans] = pos
            else:
                counts[ans] += 1

        if not counts:
            continue

        best_ans = max(counts.keys(), key=lambda a: (counts[a], -first_pos[a]))
        if _norm_answer_for_compare(best_ans) == _norm_answer_for_compare(ground_truth):
            mv_successes += 1

    return pass_successes / trials, mv_successes / trials, token_sum / trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Directory containing question_*.json files. Repeat to pool traces across runs.",
    )
    ap.add_argument(
        "--glob",
        default="question_*.json",
        help="Glob pattern under each --run-dir (default: question_*.json).",
    )
    ap.add_argument(
        "--k",
        default="256",
        help="Comma-separated k values for pass@k sampling (default: 256).",
    )
    ap.add_argument(
        "--allow-missing-questions",
        action="store_true",
        help="Skip questions absent from some run dirs instead of failing.",
    )
    ap.add_argument(
        "--suppress-parse-errors",
        action="store_true",
        help="Skip unparseable question files instead of failing.",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Sampling trials per question per k (default: 100).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (default: 0).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final overall pass@k (one line per k).",
    )
    ap.add_argument(
        "--answer-key-jsonl",
        default=None,
        help=(
            "Optional JSONL file mapping question->answer (e.g. gambit/datasets/aime_2025.jsonl). "
            "If provided, correctness is recomputed from extracted_answer vs this key (ignores stored is_trace_correct)."
        ),
    )
    args = ap.parse_args()

    run_dirs = [os.path.abspath(d) for d in args.run_dir]
    for d in run_dirs:
        if not os.path.isdir(d):
            print(f"ERROR: --run-dir is not a directory: {d}", file=sys.stderr)
            return 2

    try:
        ks = _parse_k_list(args.k)
    except Exception as e:
        print(f"ERROR parsing --k: {e}", file=sys.stderr)
        return 2

    if args.trials <= 0:
        print("ERROR: --trials must be positive", file=sys.stderr)
        return 2

    answer_key = None
    if args.answer_key_jsonl:
        try:
            answer_key = _load_answer_key_jsonl(args.answer_key_jsonl)
            if not answer_key:
                raise ValueError("answer key loaded but is empty")
        except Exception as e:
            print(f"ERROR loading --answer-key-jsonl: {e}", file=sys.stderr)
            return 2

    try:
        pools = _build_pools_impl(
            run_dirs,
            args.glob,
            allow_missing_questions=args.allow_missing_questions,
            suppress_parse_errors=args.suppress_parse_errors,
            answer_key=answer_key,
        )
    except Exception as e:
        print(f"ERROR building pools: {e}", file=sys.stderr)
        return 2

    if not pools:
        print(f"ERROR: No question files matched {args.glob!r} under run dirs", file=sys.stderr)
        return 2

    per_k_overall_sum = {k: 0.0 for k in ks}
    per_k_token_sum = {k: 0.0 for k in ks}  # sum over questions of avg_token_sum (per trial)
    per_k_majority_sum = {k: 0.0 for k in ks}

    if not args.quiet:
        print("run_dirs:")
        for d in run_dirs:
            print(f"- {d}")
        print(f"num_questions: {len(pools)}")
        print(f"trials: {args.trials}")
        print("")
        header = ["qidx", "correct/total"] + [f"pass@{k}" for k in ks]
        print("\t".join(header))
        print("-" * 80)

    # k is clamped to each question's pool size; say so once rather than
    # silently reporting pass@k computed at a smaller k.
    min_pool = min(len(qp.correct_flags) for qp in pools)
    for k in ks:
        if k > min_pool:
            print(
                f"WARNING: k={k} exceeds the smallest pool ({min_pool} traces); "
                f"pass@{k} is computed at k=pool size for those questions.",
                file=sys.stderr,
            )

    for qp in pools:
        pool_c = qp.correct_flags
        pool_t = qp.token_lengths
        pool_a = qp.extracted_answers
        correct = sum(pool_c)
        total = len(pool_c)

        row = [str(qp.question_index), f"{correct}/{total}"]
        for k in ks:
            derived_seed = (args.seed * 1000003 + qp.question_index * 9176 + k * 1315423911) & 0xFFFFFFFFFFFF
            pass_est, mv_est, tok_est = _estimate_pass_majority_and_tokens_at_k(
                pool_c,
                pool_t,
                pool_a,
                k=k,
                trials=args.trials,
                seed=int(derived_seed),
                ground_truth=qp.ground_truth,
            )
            per_k_overall_sum[k] += pass_est
            per_k_token_sum[k] += tok_est
            per_k_majority_sum[k] += mv_est
            if not args.quiet:
                row.append(f"{pass_est:.3f}")

        if not args.quiet:
            print("\t".join(row))

    if not args.quiet:
        print("-" * 80)

    for k in ks:
        overall = per_k_overall_sum[k] / len(pools)
        avg_tok = per_k_token_sum[k] / len(pools)
        print(f"overall_pass@{k}_accuracy: {overall:.6f}  (mean over {len(pools)} questions, {args.trials} trials)")
        print(f"overall_avg_sampled_token_length_sum@{k}: {avg_tok:.2f}  (mean over {len(pools)} questions, {args.trials} trials)")
        mv = per_k_majority_sum[k] / len(pools)
        print(f"overall_majority_vote_accuracy@{k}: {mv:.6f}  (mean over {len(pools)} questions, {args.trials} trials)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

