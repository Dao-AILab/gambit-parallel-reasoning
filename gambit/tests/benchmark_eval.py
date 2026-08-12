import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional

import numpy as np
import torch
from evaluator import math_equal
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if "\\text{" in text and "}" in text:
        while "\\text{" in text:
            start = text.find("\\text{")
            if start == -1:
                break
            end = text.find("}", start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting"""
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def _normalize_math_expr(s: str) -> str:
    """Normalize Unicode math characters to ASCII/LaTeX-compatible form.

    Converts Unicode √ → sqrt(), Unicode dashes → ASCII minus, etc.
    so that math_equal() can evaluate them.
    """
    import re as _re
    # Remove thousands-comma separators in pure integers (e.g. "1,234" → "1234")
    # but keep comma-separated pairs like "-13/96, 13/40" intact
    if _re.fullmatch(r'-?[\d,]+', s):
        s = s.replace(",", "")
    # Unicode minus signs → ASCII -
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    # Unicode × → *
    s = s.replace('\u00d7', '*')
    # Replace √N or √(expr) with sqrt(N) or sqrt(expr)
    # Handle √N where N is a digit sequence
    s = _re.sub(r'√(\d+)', r'sqrt(\1)', s)
    # Handle √(expr)
    s = _re.sub(r'√\(([^)]+)\)', r'sqrt(\1)', s)
    # Handle implicit multiplication: "8sqrt" → "8*sqrt", "64(" → "64*("
    s = _re.sub(r'(\d)(sqrt)', r'\1*\2', s)
    s = _re.sub(r'(\d)\(', r'\1*(', s)
    return s.strip()


# A regex fragment matching a "math expression" in Phi-4 output.
# Covers integers, decimals, fractions, radical expressions (√), Unicode dashes.
# The character class includes Unicode minus/dash variants (–, —, −).
_MATH_EXPR = r'(-?[\d√()\-\u2013\u2014\u2212+*/^. ]+(?:/[\d√()\-\u2013\u2014\u2212+*/^. ]+)?)'


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from text.

    Tries in order:
    1. LaTeX \\boxed{} (DeepSeek, Qwen, most models)
    2. Phi-4-style: explicit answer phrases after </think>
    3. Phi-4-style: explicit answer phrases in full text
    """
    import re as _re

    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()

    # Phi-4-reasoning-plus uses <think>…</think> and writes answers in plain text.
    # Focus on the post-think section when available, else use the full text.
    search_text = text.split("</think>", 1)[-1] if "</think>" in text else text

    def _extract_from(src: str) -> Optional[str]:
        # First try to extract a single-letter multiple-choice answer (A/B/C/D).
        # These patterns must be checked before the math patterns since _MATH_EXPR
        # requires at least one digit and would reject single-letter answers.
        mc_patterns = [
            r'[Ff]inal [Aa]nswer\s*[:\-]\s*\**([A-Da-d])\**\s*(?:[.!\n<]|$)',
            r'^[Aa]nswer\s*[:\-]\s*\**([A-Da-d])\**\s*\.?\s*$',
            r'[Tt]he (?:correct |final )?answer is\s*[:\-]?\s*\**([A-Da-d])\**\s*(?:[.!\n<]|$)',
            r'[Cc]orrect answer[:\s]+\**([A-Da-d])\**\s*(?:[.!\n<]|$)',
        ]
        for pat in mc_patterns:
            matches = _re.findall(pat, src, _re.MULTILINE)
            if matches:
                return matches[-1].upper()

        # Patterns ordered from most-specific to least-specific.
        # Each captures the answer expression in group 1.
        patterns = [
            # "Final Answer: 8√10" / "Final answer: 279"
            r'[Ff]inal [Aa]nswer\s*[:\-]\s*' + _MATH_EXPR,
            # "Answer: 9√15" (line-anchored to avoid matching mid-sentence)
            r'^[Aa]nswer\s*[:\-]\s*' + _MATH_EXPR + r'\s*\.?\s*$',
            # "the answer is 8√10" / "the final answer is 70"
            r'[Tt]he (?:final )?answer is\s*[:\-]?\s*' + _MATH_EXPR,
            # "Therefore, … is 279." / "So, the required difference is 279."
            r'(?:Therefore|Thus|Hence|So)[^.\n]{0,80} is\s+' + _MATH_EXPR + r'\s*[.!]?\s*$',
        ]
        for pat in patterns:
            matches = _re.findall(pat, src, _re.MULTILINE)
            if matches:
                raw = matches[-1].strip().rstrip('.')
                # Reject empty or too-long captures
                if not raw or len(raw) > 50:
                    continue
                # Normalize first (converts Unicode dashes, √, etc.)
                normalized = _normalize_math_expr(raw)
                if not normalized:
                    continue
                # Reject if result contains no digit — pure-letter strings like
                # "a" from "the answer is a 4-digit number" are not math answers.
                if not _re.search(r'\d', normalized):
                    continue
                return normalized
        return None

    result = _extract_from(search_text)
    return result


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def prepare_prompt(question: str, tokenizer) -> str:
    """Chat-template a question, asking for a \\boxed{} final answer."""
    system = "Please reason step by step, and put your final answer within \\boxed{}"
    messages = [
        {'role': "system", "content": system},
        {"role": "user", "content": question}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return full_prompt


def load_benchmark(benchmark_path: str) -> List[dict]:
    benchmark = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question")
            answer = record.get("answer")
            if question is None:
                raise ValueError(f"Missing 'question' on line {line_no}")
            benchmark.append({"question": question, "answer": answer})
    return benchmark


def ensure_output_dir(
    output_root: str, benchmark_path: str, model_path: str, num_traces: int, run_label: Optional[str] = None
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    benchmark_name = Path(benchmark_path).stem
    model_name = Path(model_path).name or Path(model_path).stem
    model_label = model_name.replace(" ", "_")
    label = run_label.strip().replace(" ", "_") if run_label else ""
    parts = [benchmark_name, model_label, f"n{num_traces}"]
    if label:
        parts.append(label)
    parts.append(timestamp)
    run_dir = os.path.join(output_root, "_".join(parts))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_single_question(
    llm: LLM,
    tokenizer,
    question: str,
    ground_truth: Optional[str],
    sampling_params: SamplingParams,
    question_index: int,
    output_dir: str,
    enable_branching: bool = False,
):
    # Important: clear Gambit state between questions so the tree export for this
    # question doesn't include completed traces from previous questions.
    if enable_branching and hasattr(llm, "reset_gambit_state"):
        llm.reset_gambit_state()
    prompt = prepare_prompt(question, tokenizer)
    bt = time()
    outputs = llm.generate([prompt], sampling_params)
    generation_time = time() - bt

    # Capture Gambit tree export after generation (if enabled)
    gambit_tree_export = None
    if enable_branching:
        gambit_tree_export = llm.get_gambit_tree_export()

    request_output = outputs[0]
    answers_for_prompt = []
    weights_for_prompt = []
    generation_results = {}
    correct_traces = 0
    total_traces = len(request_output.outputs)
    total_tokens = 0
    pruned_token_count = 0  # Tokens from pruned traces (for compute accounting)
    num_pruned_traces = 0   # Count of pruned traces

    # If Gambit is enabled and we have a tree export, treat ALL completed traces
    # (roots + branches) as "normal end-to-end traces" for voting.
    if enable_branching and isinstance(gambit_tree_export, dict):
        nodes = gambit_tree_export.get("nodes") or {}
        completed = []
        if isinstance(nodes, dict):
            for req_id, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                # Pruned traces never reach is_completed (see TraceTree.mark_pruned),
                # so this check must come first. Their tokens were still generated
                # and must be counted, but they do not vote.
                if node.get("is_pruned", False):
                    token_ids = node.get("output_token_ids") or []
                    token_count_total = int(node.get("output_token_count") or len(token_ids))
                    prefix_len = int(node.get("branch_parent_output_len") or 0)
                    pruned_token_count += max(0, token_count_total - prefix_len)
                    num_pruned_traces += 1
                    continue
                if not node.get("is_completed", False):
                    continue
                completed.append((req_id, node))

        # Deterministic order.
        completed.sort(key=lambda it: it[0])

        # Include pruned tokens in total for accurate compute accounting
        total_tokens += pruned_token_count

        total_traces = len(completed)
        for j, (req_id, node) in enumerate(completed):
            token_ids = node.get("output_token_ids") or []
            token_count_total = int(node.get("output_token_count") or len(token_ids))
            prefix_len = int(node.get("branch_parent_output_len") or 0)
            token_count_since_branch = max(0, token_count_total - prefix_len)
            total_tokens += token_count_since_branch
            score = node.get("final_score", None)

            # Decode tokens and extract the final answer.
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            extracted_answer = extract_answer(generated_text)
            is_trace_correct = (
                ground_truth is not None
                and extracted_answer is not None
                and equal_func(extracted_answer, ground_truth)
            )
            if is_trace_correct:
                correct_traces += 1
            if extracted_answer is not None:
                answers_for_prompt.append(extracted_answer)
                weights_for_prompt.append(score if score is not None else 1.0)

            # Get score history from the node
            score_history = node.get("score_history", [])

            generation_results[f"generation_{j}"] = {
                # Avoid double counting shared prefixes across branches:
                # token_length counts tokens generated since this trace's
                # branching point, while token_length_total is the full
                # end-to-end output length (excluding the original user prompt).
                "token_length": token_count_since_branch,
                "token_length_total": token_count_total,
                "token_length_prefix_inherited": prefix_len,
                "generated_text": generated_text.strip(),
                "finish_reason": node.get("finish_reason"),
                "stop_reason": node.get("stop_reason"),
                "final_score": score,
                "score_history": score_history,
                "prompt_index": question_index,
                "request_id": req_id,
                "extracted_answer": extracted_answer,
                "is_trace_correct": is_trace_correct,
                # Gambit metadata
                "gambit_is_branch": node.get("parent_id") is not None,
                "gambit_parent_id": node.get("parent_id"),
                "gambit_branch_depth": int(node.get("branch_depth") or 0),
            }

            print(
                f"question {question_index} trace {j} req_id={req_id} token_count={token_count_since_branch}, "
                f"final_score={score}, num_scores={len(score_history)}, depth={node.get('branch_depth')}, trace_correct={is_trace_correct}"
            )
    else:
        for j, output in enumerate(request_output.outputs):
            generated_text = output.text
            token_count = len(output.token_ids)
            total_tokens += token_count
            score = getattr(output, "final_score", None)
            score_history = getattr(output, "score_history", None) or []
            extracted_answer = extract_answer(generated_text)
            is_trace_correct = (
                ground_truth is not None and extracted_answer is not None and equal_func(extracted_answer, ground_truth)
            )
            if is_trace_correct:
                correct_traces += 1
            if extracted_answer is not None:
                answers_for_prompt.append(extracted_answer)
                weights_for_prompt.append(score if score is not None else 1.0)

            generation_results[f"generation_{j}"] = {
                "token_length": token_count,
                "generated_text": generated_text.strip(),
                "finish_reason": output.finish_reason,
                "stop_reason": output.stop_reason,
                "final_score": score,
                "score_history": score_history,
                "prompt_index": question_index,
                "request_id": getattr(request_output, "request_id", None),
                "extracted_answer": extracted_answer,
                "is_trace_correct": is_trace_correct,
                # Gambit branch metadata
                "gambit_is_branch": getattr(output, "gambit_is_branch", False),
                "gambit_parent_id": getattr(output, "gambit_parent_id", None),
                "gambit_branch_depth": getattr(output, "gambit_branch_depth", 0),
            }

            print(
                f"question {question_index} output {j} token count: {token_count}, "
                f"finish_reason={output.finish_reason}, stop_reason={output.stop_reason}, final_score={score}, "
                f"num_scores={len(score_history)}, trace_correct={is_trace_correct}"
            )

    final_answer = weighted_majority_vote(answers_for_prompt, weights_for_prompt) if answers_for_prompt else None
    is_final_correct = (
        final_answer is not None and ground_truth is not None and equal_func(final_answer, ground_truth)
    )

    # Calculate Gambit statistics (prefer tree export if present)
    if enable_branching and isinstance(gambit_tree_export, dict):
        nodes = gambit_tree_export.get("nodes") or {}
        if isinstance(nodes, dict) and nodes:
            num_branches = sum(
                1 for node in nodes.values()
                if isinstance(node, dict) and node.get("parent_id") is not None
            )
            max_branch_depth = max(
                (
                    int(node.get("branch_depth") or 0)
                    for node in nodes.values()
                    if isinstance(node, dict)
                ),
                default=0,
            )
        else:
            num_branches = sum(1 for g in generation_results.values() if g.get("gambit_is_branch", False))
            max_branch_depth = max((g.get("gambit_branch_depth", 0) for g in generation_results.values()), default=0)
    else:
        num_branches = sum(1 for g in generation_results.values() if g.get("gambit_is_branch", False))
        max_branch_depth = max((g.get("gambit_branch_depth", 0) for g in generation_results.values()), default=0)

    question_result = {
        "question_index": question_index,
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "is_final_correct": is_final_correct,
        "num_correct_traces": correct_traces,
        "total_traces": total_traces,
        "generation_time": generation_time,
        "answers_for_prompt": answers_for_prompt,
        "weights_for_prompt": weights_for_prompt,
        "generation_results": generation_results,
        # Gambit tree info (if enabled)
        "gambit_enabled": enable_branching,
        "gambit_num_branches": num_branches,
        "gambit_max_branch_depth": max_branch_depth,
        "gambit_tree_export": gambit_tree_export,
    }

    question_file = os.path.join(output_dir, f"question_{question_index:04d}.json")
    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_result, f, ensure_ascii=False, indent=2)

    return {
        "question_index": question_index,
        "question_file": os.path.basename(question_file),
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "is_final_correct": is_final_correct,
        "num_correct_traces": correct_traces,
        "total_traces": total_traces,
        "correct_trace_ratio": correct_traces / total_traces if total_traces else 0.0,
        "generation_time": generation_time,
        "total_tokens": total_tokens,
        # Token breakdown for compute accounting
        "pruned_tokens": pruned_token_count,
        "completed_tokens": total_tokens - pruned_token_count,
        "num_pruned_traces": num_pruned_traces,
        # Gambit stats
        "gambit_num_branches": num_branches,
        "gambit_max_branch_depth": max_branch_depth,
    }


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark inference over a JSONL dataset.")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to JSONL benchmark file with 'question' and 'answer' fields (e.g., gambit/datasets/hmmt_2024.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory to store per-question outputs and the summary.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional custom label appended to the generated output directory name.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the vLLM model weights.",
    )
    parser.add_argument(
        "--gambit-step-scorer-path",
        required=True,
        help="Path to the step scorer checkpoint.",
    )
    parser.add_argument("--num-traces", type=int, default=256, help="Number of generations per question.")
    parser.add_argument("--max-tokens", type=int, default=60000, help="Maximum new tokens per generation.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to reserve for the LLM (vLLM gpu_memory_utilization).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help=(
            "Override vLLM max_model_len (positional context size used to "
            "size KV cache during profile_run). 0 = use model default. Set to "
            "a smaller value (e.g. 32000) on shared GPUs where the default "
            "262144 context would force a negative KV cache budget."
        ),
    )
    # Gambit arguments
    parser.add_argument(
        "--enable-branching",
        action="store_true",
        help="Enable Gambit branching to dynamically branch high-scoring traces.",
    )
    # Tournament mode arguments (recommended)
    parser.add_argument(
        "--tournament-mode",
        action="store_true",
        default=True,
        help="Use tournament-style selection (rank-based, recommended). Set --no-tournament-mode to disable.",
    )
    parser.add_argument(
        "--no-tournament-mode",
        action="store_false",
        dest="tournament_mode",
        help="Disable tournament mode and use threshold-based selection.",
    )
    parser.add_argument(
        "--tournament-capacity",
        type=int,
        default=256,
        help="Maximum active traces in tournament mode (C).",
    )
    parser.add_argument(
        "--tournament-swap-k",
        type=int,
        default=16,
        help="Number of traces to swap in each tournament round (K).",
    )
    parser.add_argument(
        "--tournament-check-interval",
        type=int,
        default=200,
        help="Scoring/decision rounds between tournament rounds.",
    )
    parser.add_argument(
        "--tournament-warmup-tokens",
        type=int,
        default=12000,
        help="Minimum tokens before tournament starts.",
    )
    parser.add_argument(
        "--tournament-hard-floor",
        type=float,
        default=0.1,
        help="Kill any trace below this score (garbage collection).",
    )
    parser.add_argument(
        "--tournament-branch-random",
        action="store_true",
        help="ABLATION: branch on K randomly-sampled active traces instead of "
             "the score-top-K. Pruning still uses bottom-K by score.",
    )
    parser.add_argument(
        "--scoring-stride",
        type=int,
        default=0,
        help=(
            "Trigger hidden-state capture every N generated tokens instead of "
            "at \\n\\n boundaries. 0 (default) = use \\n\\n detection. "
            "Recommended: 32 for models with coarse \\n\\n granularity "
            "(Qwen3-14B, Phi-4, Qwen3-4B)."
        ),
    )
    parser.add_argument(
        "--stop-after-completed-traces",
        type=int,
        default=0,
        help=(
            "If > 0, stop generating for the current question once this many "
            "traces (roots + branches) have completed. Remaining active traces "
            "will be stopped early."
        ),
    )
    parser.add_argument(
        "--no-score-history",
        action="store_true",
        help=(
            "Disable recording score_history (saves RAM). Tournament/threshold "
            "decisions still use running averages, but outputs/tree exports will "
            "not include per-step score trajectories."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and CUDA graphs (useful for unsupported GPU architectures).",
    )
    # Legacy threshold mode arguments
    parser.add_argument(
        "--theta-branch",
        type=float,
        default=0.75,
        help="(Threshold mode) Score threshold for branching (branch if score > this).",
    )
    parser.add_argument(
        "--theta-prune",
        type=float,
        default=0.15,
        help="(Threshold mode) Score threshold for proactive pruning (prune if score < this).",
    )
    parser.add_argument(
        "--max-total-traces",
        type=int,
        default=64,
        help="(Threshold mode) Maximum total traces budget for Gambit.",
    )
    parser.add_argument(
        "--enable-gambit",
        action="store_true",
        help="Enable Gambit.",
    )
    parser.add_argument(
        "--problem-indices",
        type=str,
        default="",
        help="Comma-separated problem indices to run (default: all problems).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Run-level random seed. Threaded into SamplingParams (vLLM), and used "
            "to seed Python/NumPy/PyTorch RNGs. Per-question SamplingParams.seed is "
            "set to (seed + question_idx) so each question is deterministic while "
            "still differing across seeds."
        ),
    )
    args = parser.parse_args()

    # Seed all Python/NumPy/PyTorch RNGs from the run-level seed. CUDA seeding is
    # best-effort: vLLM v1 spawns worker processes that re-seed independently;
    # the per-question SamplingParams(seed=...) below is what actually controls
    # generation determinism.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Phi-4-reasoning-plus has a 32K context window, so it cannot honour the
    # 60K default. Clamp to the model's capacity rather than failing at runtime.
    if 'Phi' in args.model_path and args.max_tokens > 32000:
        args.max_tokens = 32000
        print(f"[max-tokens] clamped to {args.max_tokens} for {args.model_path}")

    benchmark = load_benchmark(args.benchmark)
    output_dir = ensure_output_dir(
        args.output_dir, args.benchmark, args.model_path, args.num_traces, run_label=args.run_label
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    # Per-question SamplingParams is constructed inside the question loop so that
    # we can set seed = args.seed + question_idx (deterministic per question,
    # differentiable across seeds). Stash the shared kwargs here.
    # Fail here rather than after the model is loaded and generating.
    if args.enable_branching and not args.enable_gambit:
        raise SystemExit(
            "--enable-branching requires --enable-gambit: without it no step "
            "scores are produced, so the tournament never runs and the job "
            "silently degrades to plain parallel sampling."
        )
    if args.enable_gambit and not args.gambit_step_scorer_path:
        raise SystemExit("--enable-gambit requires --gambit-step-scorer-path.")
    if args.enable_gambit and not os.path.exists(args.gambit_step_scorer_path):
        raise SystemExit(f"Scorer checkpoint not found: {args.gambit_step_scorer_path}")

    _sampling_kwargs = dict(
        n=args.num_traces,
        temperature=0.6 if 'Phi' not in args.model_path else 0.8,
        top_p=0.95,
        top_k=20 if 'Phi' not in args.model_path else 50,
        max_tokens=args.max_tokens,
    )
    _llm_extra_kwargs = {}
    if getattr(args, "max_model_len", 0) and args.max_model_len > 0:
        _llm_extra_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **_llm_extra_kwargs,
        gambit_enable=args.enable_gambit,
        gambit_step_scorer_path=args.gambit_step_scorer_path,
        gambit_scoring_stride=args.scoring_stride,
        gambit_record_score_history=(not args.no_score_history),
        gambit_enable_branching=args.enable_branching,
        gambit_stop_after_completed_traces=args.stop_after_completed_traces,
        # Tournament mode parameters
        gambit_tournament_mode=args.tournament_mode,
        gambit_tournament_capacity=args.tournament_capacity,
        gambit_tournament_swap_k=args.tournament_swap_k,
        gambit_tournament_check_interval=args.tournament_check_interval,
        gambit_tournament_warmup_tokens=args.tournament_warmup_tokens,
        gambit_tournament_hard_floor=args.tournament_hard_floor,
        gambit_tournament_branch_random=args.tournament_branch_random,
        # Legacy threshold mode parameters
        gambit_theta_branch=args.theta_branch,
        gambit_theta_prune=args.theta_prune,
        gambit_max_total_traces=args.max_total_traces,
        disable_log_stats=False,
        enforce_eager=getattr(args, 'enforce_eager', False),
    )

    # Log Gambit configuration if enabled
    if args.enable_branching:
        if args.tournament_mode:
            print(f"Gambit Tournament mode: capacity={args.tournament_capacity}, "
                  f"swap_k={args.tournament_swap_k}, check_interval={args.tournament_check_interval}, "
                  f"warmup={args.tournament_warmup_tokens}, hard_floor={args.tournament_hard_floor}, "
                  f"stop_after_completed_traces={args.stop_after_completed_traces}, "
                  f"record_score_history={not args.no_score_history}, ")
        else:
            print(f"Gambit Threshold mode: theta_branch={args.theta_branch}, "
                  f"theta_prune={args.theta_prune}, max_total_traces={args.max_total_traces}, "
                  f"stop_after_completed_traces={args.stop_after_completed_traces}, "
                  f"record_score_history={not args.no_score_history}")

    summary_entries = []

    total_start = time()
    if args.problem_indices:
        explicit_indices = set(int(x) for x in args.problem_indices.split(",") if x.strip())
        print(f"[problem-indices] Running only indices: {sorted(explicit_indices)}")
    else:
        explicit_indices = None

    for idx in range(len(benchmark)):
        if explicit_indices is not None and idx not in explicit_indices:
            continue

        item = benchmark[idx]
        question = item["question"]
        ground_truth = item.get("answer")
        print(f"Running question {idx + 1}/{len(benchmark)}")
        # Per-question seed: differentiates seeds across runs while keeping a
        # given (run-seed, question) pair reproducible.
        sampling_params = SamplingParams(seed=args.seed + idx, **_sampling_kwargs)
        summary_entry = run_single_question(
            llm=llm,
            tokenizer=tokenizer,
            question=question,
            ground_truth=ground_truth,
            sampling_params=sampling_params,
            question_index=idx,
            output_dir=output_dir,
            enable_branching=args.enable_branching,
        )
        summary_entries.append(summary_entry)
    total_generation_time = time() - total_start

    total_questions = len(summary_entries)
    avg_tokens_per_problem = (
        sum(entry["total_tokens"] for entry in summary_entries) / total_questions if total_questions else 0.0
    )
    final_accuracy = (
        sum(1 for entry in summary_entries if entry["is_final_correct"]) / total_questions if total_questions else 0.0
    )

    # Calculate Gambit aggregate statistics
    total_branches = sum(entry.get("gambit_num_branches", 0) for entry in summary_entries)
    avg_branches_per_question = total_branches / total_questions if total_questions else 0.0
    max_branch_depth_overall = max(
        (entry.get("gambit_max_branch_depth", 0) for entry in summary_entries), default=0
    )

    summary = {
        "benchmark_path": args.benchmark,
        "model_path": args.model_path,
        "output_dir": output_dir,
        "num_questions": total_questions,
        "num_traces_per_question": args.num_traces,
        "seed": args.seed,
        "total_generation_time": total_generation_time,
        "generation_time_per_question": total_generation_time / total_questions if total_questions else 0.0,
        "final_accuracy": final_accuracy,
        "avg_tokens_per_problem": avg_tokens_per_problem,
        # Gambit configuration and stats
        "gambit_config": {
            "enabled": args.enable_branching,
            "tournament_mode": args.tournament_mode,
            # Tournament mode params
            "tournament_capacity": args.tournament_capacity,
            "tournament_swap_k": args.tournament_swap_k,
            "tournament_check_interval": args.tournament_check_interval,
            "tournament_warmup_tokens": args.tournament_warmup_tokens,
            "tournament_hard_floor": args.tournament_hard_floor,
            "stop_after_completed_traces": args.stop_after_completed_traces,
            # Threshold mode params
            "theta_branch": args.theta_branch,
            "theta_prune": args.theta_prune,
            "max_total_traces": args.max_total_traces,
        },
        "gambit_stats": {
            "total_branches": total_branches,
            "avg_branches_per_question": avg_branches_per_question,
            "max_branch_depth": max_branch_depth_overall,
            "total_pruned_traces": sum(entry.get("num_pruned_traces", 0) for entry in summary_entries),
            "total_pruned_tokens": sum(entry.get("pruned_tokens", 0) for entry in summary_entries),
            "total_completed_tokens": sum(entry.get("completed_tokens", 0) for entry in summary_entries),
        },
        "questions": summary_entries,
    }

    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved per-question outputs to {output_dir}")
    print(f"Total generation time: {total_generation_time:.2f}s")
    print(f"Generation time per question: {total_generation_time / total_questions if total_questions else 0.0:.2f}s")
    print(f"Final benchmark accuracy: {final_accuracy:.4f}")
    print(f"Avg tokens per problem: {avg_tokens_per_problem:.1f}")
    if args.enable_branching:
        total_pruned_traces = sum(entry.get("num_pruned_traces", 0) for entry in summary_entries)
        total_pruned_tokens = sum(entry.get("pruned_tokens", 0) for entry in summary_entries)
        total_completed_tokens = sum(entry.get("completed_tokens", 0) for entry in summary_entries)
        print(f"Gambit Stats: total_branches={total_branches}, "
              f"avg_branches_per_question={avg_branches_per_question:.2f}, "
              f"max_branch_depth={max_branch_depth_overall}")
        print(f"Token breakdown: completed={total_completed_tokens}, "
              f"pruned={total_pruned_tokens} ({total_pruned_traces} traces), "
              f"total={total_completed_tokens + total_pruned_tokens}")



if __name__ == "__main__":
    main()
