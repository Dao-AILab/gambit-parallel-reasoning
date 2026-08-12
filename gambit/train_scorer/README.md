# Training the history-aware sequence scorer

Gambit's tournament is agnostic to the scoring function. Our main evaluation uses
an off-the-shelf 2-layer MLP step scorer (checkpoints in
[`gambit/step_scorer_checkpoint`](../step_scorer_checkpoint)), which scores each
reasoning step in isolation.

This directory contains the recipe for the **history-aware sequence scorer**
(paper, Appendix A.1) — an alternative guidance signal used to study how scorer
quality affects search dynamics. Instead of scoring a step in isolation, it maps
a *prefix* of per-step hidden states to a quality score in `[0, 1]` by attending
over the full reasoning trajectory:

```
h~_t     = GELU(W_in · LN(h_t))
z_1..z_T = SequenceTransformer(h~_1, ..., h~_T)
y^_t     = sigmoid(w_out^T · LN(z_t))
```

Each layer is a pre-norm block with multi-head **causal** self-attention using
RoPE and a SwiGLU feed-forward network. The causal mask is what makes the scorer
history-aware: position `t` attends to steps `1..t`, so `y^_t` is conditioned on
the entire reasoning history up to step `t`. This lets it flag globally inferior
steps that look locally plausible.

## Pipeline

Two steps: extract hidden states, then train.

### 1. Extract step-boundary hidden states

Generate `n` independent traces per problem with standard parallel sampling and
save them as JSONL, one object per line:

```json
{"question": "...", "response": "<think>...</think> ... \\boxed{42}", "ground_truth": "42"}
```

`response` is the full generation including the `<think>` block. An
`extracted_answer` field is used if present; otherwise the answer is re-extracted
from `response`. [`train_problems/hmmt_2012_2023.jsonl`](train_problems/hmmt_2012_2023.jsonl)
provides 357 additional `{question, answer}` problems to sample traces from,
beyond the benchmark sets in [`gambit/datasets`](../datasets).

```bash
python gambit/train_scorer/extract_hidden_states.py \
  --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --input_path traces.jsonl \
  --output_dir hidden_states/train \
  --shard_size 1000
```

Reasoning steps are delimited by double newlines (`\n\n`) inside the `<think>`
block. For each boundary, the script records the last-layer hidden state at the
token that *contains* the `\n\n` — not the one before it. Tokenizers routinely
merge the delimiter with adjacent text (e.g. `".\n\n"`), and the engine captures
on that same containing token, so this is what keeps training data aligned with
inference-time capture. This yields a per-trace step sequence `(h_1, ..., h_T)`.
The trace is labelled with the binary correctness of its final extracted answer.
Output is sharded `.pt` files; run it once per split.

> The model must be the same one that generated the traces — hidden states are
> model-specific, and the tokenizer must reproduce the exact generated text.

### 2. Train the scorer

```bash
python gambit/train_scorer/train_sequence_scorer.py \
  --train_dir hidden_states/train \
  --test_dir  hidden_states/test \
  --d_model 256 --nhead 4 --num_layers 1 --dropout 0.1 --max_len 4096 \
  --epochs 30 --batch_size 8 --lr 1e-4 \
  --config_name seq_d256_1l
```

**Training objective — last-step BCE.** Each trace is a single sample: the full
sequence `[h_1..h_T]` is fed in, but the loss is taken *only* at the last valid
position `T`, against the trace-level correctness label `y`:

```
L(θ) = -1/|D| · Σ  [ y·log y^_T + (1-y)·log(1-y^_T) ]
```

Only the final position contributes a gradient, which forces the attention
mechanism to credit-assign over the whole trajectory rather than lean on local
features at any single step. It also matches inference, where the engine reads
the logit at the final observed step.

Training uses class-weighted BCE (`pos_weight = n_neg/n_pos`), AdamW with cosine
decay after a linear warm-up, gradient clipping, and early stopping on
`0.75·AUC + 0.25·F1` measured with last-step scoring on the validation split.

Checkpoints are written to `./checkpoints/<config_name>/` and embed a
`model_config` dict, so inference reconstructs the architecture automatically.

Useful knobs:

| flag | default | notes |
|---|---|---|
| `--d_model` / `--nhead` / `--num_layers` | 256 / 4 / 1 | ~1.6M params at the default |
| `--rope_base` | 10000 | step sequences are short; 300–500 gives finer angular resolution |
| `--max_len` | 4096 | longer traces are truncated |
| `--val_split` | 0.15 | always carved out of `--train_dir`; `--test_dir` adds a separate held-out report, it does not replace validation |

## Using a trained scorer

Pass the checkpoint to the engine like any other scorer:

```bash
--gambit-step-scorer-path checkpoints/seq_d256_1l/checkpoint_....pt
```

The engine detects a sequence scorer from the checkpoint's
`model_config["type"] == "sequence_transformer"` and maintains the per-request
hidden-state history needed for causal scoring, including copying the history on
branch and clearing it on prune.
