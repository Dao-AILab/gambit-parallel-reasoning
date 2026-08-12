# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pydantic.dataclasses import dataclass
from .utils import config

@config
@dataclass
class GambitConfig:
    # Opt-in: with this False the fork behaves exactly like upstream vLLM.
    # Enabling it requires a tokenizer (see set_gambit_token_ids), so it is not
    # supported on the AsyncLLM / `vllm serve` path.
    enable: bool = False
    step_scorer_path: str | None = None
    stop_thinking_tokenID: int | None = None
    double_new_line_tokenID: list[int] | None = None
    # Fixed-stride scoring: trigger hidden-state capture every N generated
    # tokens instead of (or in addition to) \n\n detection.  0 = disabled
    # (only \n\n triggers capture, the default).  Recommended: 32.
    scoring_stride: int = 0
    # If False, do not store per-step score histories (to save RAM).
    # Tournament/threshold decisions will still use running averages.
    record_score_history: bool = True
    # If True, persist per-trace hidden states for on-policy mining.
    # Hidden states are copied to CPU-side Python lists to avoid GPU memory
    # growth and can be exported after each generate() call.
    collect_hidden_states_for_mining: bool = False

    # Gambit parameters
    enable_branching: bool = False  # Enable Gambit branching

    # Tournament mode (recommended) - rank-based selection instead of thresholds
    # Defaults match the configuration reported in the paper (Section 5.1).
    tournament_mode: bool = True  # Use tournament-style selection
    tournament_capacity: int = 256  # Maximum active traces (C)
    tournament_swap_k: int = 16  # Number of traces to swap when at capacity (K)
    tournament_check_interval: int = 200  # Scoring/decision rounds between tournaments
    tournament_warmup_tokens: int = 12000  # Minimum tokens before a trace may branch
    tournament_hard_floor: float = 0.1  # Kill any trace below this score
    # ABLATION: branch on K RANDOMLY-sampled active traces instead of top-K
    # by score. Pruning still uses bottom-K by score. Used to test whether
    # the "score-guided" branching is what makes Gambit work.
    tournament_branch_random: bool = False
    # If > 0, stop generating for the current question once this many traces
    # have completed (roots + branches). Remaining active traces will be stopped
    # early to avoid wasting tokens on long, low-value completions.
    stop_after_completed_traces: int = 0

    # Legacy threshold-based mode (used when tournament_mode=False)
    theta_branch: float = 0.75  # Score threshold for branching (branch if score > this)
    theta_prune: float = 0.15  # Score threshold for proactive pruning (prune if score < this)
    max_branches_per_trace: int = 2  # Maximum number of branches per fork (k)
    max_total_traces: int = 64  # Maximum total traces budget (N)
    branch_cooldown_steps: int = 200  # Minimum steps between branches for same trace
    branch_temperature_boost: float = 1.0  # Temperature multiplier for branched traces
    min_tokens_before_branch: int = 4000  # Minimum tokens before allowing branch
    min_tokens_before_prune: int = 4000  # Minimum tokens before allowing proactive prune

    def compute_hash(self) -> str:
        return str((
            self.enable,
            self.step_scorer_path,
            self.stop_thinking_tokenID,
            self.double_new_line_tokenID,
            self.scoring_stride,
            self.record_score_history,
            self.collect_hidden_states_for_mining,
            self.enable_branching,
            self.tournament_mode,
            self.tournament_capacity,
            self.tournament_swap_k,
            self.tournament_check_interval,
            self.tournament_warmup_tokens,
            self.tournament_hard_floor,
            self.tournament_branch_random,
            self.stop_after_completed_traces,
            self.theta_branch,
            self.theta_prune,
            self.max_branches_per_trace,
            self.max_total_traces,
            self.branch_cooldown_steps,
            self.branch_temperature_boost,
            self.min_tokens_before_branch,
            self.min_tokens_before_prune,
        ))

def set_gambit_token_ids(step_config, tokenizer):
    target = "\n\n"
    matching_tokens = []
    for token_id in range(tokenizer.vocab_size):
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if target in token_text:
            matching_tokens.append((token_id, token_text))
    step_config.double_new_line_tokenID = [token_id for token_id, _ in matching_tokens]
    step_config.stop_thinking_tokenID = tokenizer.added_tokens_encoder.get("</think>", None)
