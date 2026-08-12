# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TraceTree: the logical topology of Gambit's thought-level beam search.

This is the "tree view" of Section 4.2: it tracks parent/child relationships,
per-trace score histories, and completion status for every trace the search
considers active — including "ghost" traces that the scheduler has evicted
under memory pressure but that remain logically active until pruned.
"""

from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TraceNode:
    """Represents a single trace in the tree."""
    req_id: str
    parent_id: Optional[str] = None
    branch_depth: int = 0
    created_at_step: int = 0
    last_branch_step: int = -1  # Step when this trace last branched
    # For branches, how many "output tokens" were already present at the moment
    # this branch was created (i.e., the inherited prefix length). This lets
    # callers compute token counts "since branch" without double-counting
    # ancestor output across siblings.
    branch_parent_output_len: int = 0
    score_history: list[float] = field(default_factory=list)
    is_active: bool = True
    is_completed: bool = False
    is_pruned: bool = False
    children: list[str] = field(default_factory=list)
    # Output data (populated when trace completes)
    output_text: Optional[str] = None
    output_token_ids: Optional[list[int]] = None
    output_token_count: int = 0
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    final_score: Optional[float] = None

    @property
    def latest_score(self) -> Optional[float]:
        """Get the most recent score for this trace."""
        if not self.score_history:
            return None
        return self.score_history[-1]

    @property
    def avg_score(self) -> Optional[float]:
        """Get the average score for this trace."""
        if not self.score_history:
            return None
        return sum(self.score_history) / len(self.score_history)

    def record_score(self, score: float) -> None:
        """Record a new score for this trace."""
        self.score_history.append(score)

    def set_output(
        self,
        output_text: Optional[str] = None,
        output_token_ids: Optional[list[int]] = None,
        finish_reason: Optional[str] = None,
        stop_reason: Optional[str] = None,
        final_score: Optional[float] = None,
    ) -> None:
        """Set the output data when trace completes."""
        self.output_text = output_text
        self.output_token_ids = output_token_ids
        self.output_token_count = len(output_token_ids) if output_token_ids else 0
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.final_score = final_score


class TraceTree:
    """
    Manages the tree of branched traces for Gambit algorithm.

    Tracks parent-child relationships, scores, and provides utilities
    for branching decisions.
    """

    def __init__(
        self,
        max_traces: int,
        branch_cooldown_steps: int = 2,
        min_tokens_before_branch: int = 10,
    ):
        """
        Initialize the trace tree.

        Args:
            max_traces: Maximum number of active traces allowed (budget N)
            branch_cooldown_steps: Minimum steps between branches for same trace
            min_tokens_before_branch: Minimum tokens generated before branching allowed
        """
        self.max_traces = max_traces
        self.branch_cooldown_steps = branch_cooldown_steps
        self.min_tokens_before_branch = min_tokens_before_branch

        # req_id -> TraceNode
        self.nodes: dict[str, TraceNode] = {}

        # Track root traces (original problem traces)
        self.root_ids: set[str] = set()

        # Counter for generating unique branch IDs
        self.branch_counter: int = 0

        # Current step counter
        self.current_step: int = 0

        # Parent request ID -> set of all descendant request IDs (for aggregation)
        self._lineage_cache: dict[str, set[str]] = {}

    def add_root(self, req_id: str) -> TraceNode:
        """Add a root trace (initial trace for a problem)."""
        node = TraceNode(
            req_id=req_id,
            parent_id=None,
            branch_depth=0,
            created_at_step=self.current_step,
        )
        self.nodes[req_id] = node
        self.root_ids.add(req_id)
        self._invalidate_lineage_cache()
        logger.debug("[Gambit] Added root trace: %s", req_id)
        return node

    def add_branch(
        self,
        parent_req_id: str,
        child_req_id: str,
        *,
        branch_parent_output_len: int = 0,
    ) -> Optional[TraceNode]:
        """
        Add a branched trace as a child of the parent.

        Args:
            parent_req_id: Request ID of the parent trace
            child_req_id: Request ID of the new branched trace

        Returns:
            The new TraceNode, or None if parent not found
        """
        parent = self.nodes.get(parent_req_id)
        if parent is None:
            logger.warning("[Gambit] Parent trace not found: %s", parent_req_id)
            return None

        child = TraceNode(
            req_id=child_req_id,
            parent_id=parent_req_id,
            branch_depth=parent.branch_depth + 1,
            created_at_step=self.current_step,
            branch_parent_output_len=branch_parent_output_len,
            # Copy parent's score history to child
            score_history=parent.score_history.copy(),
        )

        self.nodes[child_req_id] = child
        parent.children.append(child_req_id)
        parent.last_branch_step = self.current_step
        self.branch_counter += 1
        self._invalidate_lineage_cache()

        logger.debug(
            "[Gambit] Created branch: %s from parent %s (depth=%d)",
            child_req_id, parent_req_id, child.branch_depth,
        )
        return child

    def mark_completed(
        self,
        req_id: str,
        output_text: Optional[str] = None,
        output_token_ids: Optional[list[int]] = None,
        finish_reason: Optional[str] = None,
        stop_reason: Optional[str] = None,
        final_score: Optional[float] = None,
    ) -> None:
        """Mark a trace as completed (finished generation) with output data."""
        if node := self.nodes.get(req_id):
            node.is_active = False
            node.is_completed = True
            node.set_output(
                output_text=output_text,
                output_token_ids=output_token_ids,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                final_score=final_score,
            )
            logger.debug("[Gambit] Trace completed: %s (tokens=%d)", req_id, node.output_token_count)

    def mark_pruned(self, req_id: str) -> None:
        """Mark a trace as pruned."""
        if node := self.nodes.get(req_id):
            node.is_active = False
            node.is_pruned = True
            logger.debug("[Gambit] Trace pruned: %s", req_id)

    def get_active_count(self) -> int:
        """Get the number of currently active traces."""
        return sum(1 for node in self.nodes.values() if node.is_active)

    def get_completed_count(self) -> int:
        """Get the number of completed traces."""
        return sum(1 for node in self.nodes.values() if node.is_completed)

    def can_branch(self) -> bool:
        """Check if we have budget to create more branches."""
        return self.get_active_count() < self.max_traces


    def can_trace_branch(
        self,
        req_id: str,
        num_tokens_generated: int,
    ) -> bool:
        """
        Check if a specific trace can branch.

        Args:
            req_id: Request ID of the trace
            num_tokens_generated: Number of tokens generated by this trace

        Returns:
            True if the trace can branch, False otherwise
        """
        node = self.nodes.get(req_id)
        if node is None or not node.is_active:
            return False

        # Check budget
        if not self.can_branch():
            return False

        # Check minimum tokens
        if num_tokens_generated < self.min_tokens_before_branch:
            return False

        # Check cooldown
        if node.last_branch_step >= 0:
            steps_since_branch = self.current_step - node.last_branch_step
            if steps_since_branch < self.branch_cooldown_steps:
                return False

        return True

    def record_score(self, req_id: str, score: float) -> None:
        """Record a score for a trace."""
        if node := self.nodes.get(req_id):
            node.record_score(score)

    def get_score(self, req_id: str) -> Optional[float]:
        """Get the average score for a trace."""
        if node := self.nodes.get(req_id):
            return node.avg_score
        return None

    def get_node(self, req_id: str) -> Optional[TraceNode]:
        """Get the TraceNode for a request ID."""
        return self.nodes.get(req_id)

    def increment_step(self) -> None:
        """Increment the step counter."""
        self.current_step += 1




    def generate_branch_id(self, parent_req_id: str) -> str:
        """Generate a unique request ID for a branched trace."""
        return f"gambit_{self.branch_counter}_{parent_req_id}"

    def remove_trace(self, req_id: str) -> None:
        """Remove a trace from the tree (cleanup)."""
        if req_id in self.nodes:
            del self.nodes[req_id]
            self.root_ids.discard(req_id)
            self._invalidate_lineage_cache()

    def _invalidate_lineage_cache(self) -> None:
        """Invalidate the lineage cache when tree structure changes."""
        self._lineage_cache.clear()

    def get_stats(self) -> dict:
        """Get statistics about the trace tree."""
        active_count = 0
        completed_count = 0
        pruned_count = 0
        max_depth = 0

        for node in self.nodes.values():
            if node.is_active:
                active_count += 1
            if node.is_completed:
                completed_count += 1
            if node.is_pruned:
                pruned_count += 1
            max_depth = max(max_depth, node.branch_depth)

        return {
            "total_nodes": len(self.nodes),
            "active_count": active_count,
            "completed_count": completed_count,
            "pruned_count": pruned_count,
            "max_depth": max_depth,
            "num_roots": len(self.root_ids),
            "branch_counter": self.branch_counter,
            "current_step": self.current_step,
        }

    def export_tree(self) -> dict:
        """
        Export the entire tree structure as a serializable dictionary.

        Returns a dictionary with:
        - stats: Overall tree statistics
        - nodes: Dict of all nodes with their metadata and outputs
        - roots: List of root request IDs
        - completed_traces: List of completed trace data with outputs
        """
        nodes_data = {}
        completed_traces = []

        for req_id, node in self.nodes.items():
            node_data = {
                "req_id": node.req_id,
                "parent_id": node.parent_id,
                "branch_depth": node.branch_depth,
                "created_at_step": node.created_at_step,
                "last_branch_step": node.last_branch_step,
                "branch_parent_output_len": node.branch_parent_output_len,
                "score_history": node.score_history,
                "latest_score": node.latest_score,
                "avg_score": node.avg_score,
                "is_active": node.is_active,
                "is_completed": node.is_completed,
                "is_pruned": node.is_pruned,
                "children": node.children,
                # Output data
                "output_token_count": node.output_token_count,
                # Include token ids so callers can reconstruct generated text.
                # (We intentionally do not include output_text here to avoid huge
                # JSON; see completed_traces below.)
                "output_token_ids": node.output_token_ids,
                "finish_reason": node.finish_reason,
                "stop_reason": node.stop_reason,
                "final_score": node.final_score,
            }
            nodes_data[req_id] = node_data

            # Add to completed traces list with full output text
            if node.is_completed and node.output_text is not None:
                completed_traces.append({
                    "req_id": node.req_id,
                    "parent_id": node.parent_id,
                    "branch_depth": node.branch_depth,
                    "output_text": node.output_text,
                    "output_token_count": node.output_token_count,
                    "final_score": node.final_score,
                    "finish_reason": node.finish_reason,
                    "stop_reason": node.stop_reason,
                })

        return {
            "stats": self.get_stats(),
            "nodes": nodes_data,
            "roots": list(self.root_ids),
            "completed_traces": completed_traces,
        }

    def get_node_metadata(self, req_id: str) -> Optional[dict]:
        """Get metadata for a specific node as a dictionary."""
        node = self.nodes.get(req_id)
        if node is None:
            return None

        return {
            "parent_id": node.parent_id,
            "branch_depth": node.branch_depth,
            "is_branch": node.parent_id is not None,
            "branch_parent_output_len": node.branch_parent_output_len,
            "score_history": node.score_history,
            "avg_score": node.avg_score,
            "is_completed": node.is_completed,
            "is_pruned": node.is_pruned,
        }

    def __contains__(self, req_id: str) -> bool:
        return req_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)