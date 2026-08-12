# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import random
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Sequence, TypeVar, cast
import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext
from vllm.version import __version__ as VLLM_VERSION

import json
from copy import copy
from vllm.v1.engine.trace_tree import TraceTree

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if (
            self.vllm_config.cache_config.enable_prefix_caching
            or kv_connector is not None
        ):
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )

        self.gambit_config = vllm_config.gambit_config
        if self.gambit_config.enable and self.batch_queue is not None:
            # step_with_batch_queue has no scoring or tournament hooks, so Gambit
            # would silently degrade to plain parallel sampling.
            raise ValueError(
                "Gambit is not supported with the batch-queue execution path "
                "(e.g. pipeline parallelism). Run with a single execution batch "
                "or disable Gambit."
            )
        # Tracking for trace scoring/pruning
        self.trace_scores: dict[str, list[float]] = {}
        # If gambit_config.record_score_history is False, keep only running
        # aggregates to bound memory usage.
        self.trace_score_sums: dict[str, float] = {}
        self.trace_score_counts: dict[str, int] = {}
        self.final_trace_scores: dict[str, float | None] = {}
        self.pending_low_score_stops: set[str] = set()
        # Gambit: forced early-stop (e.g., stop after enough traces completed).
        # We keep this separate from low-score pruning so the stop_reason can
        # reflect why EOS was inserted.
        self.pending_force_stops: set[str] = set()
        self.trace_token_counts: dict[str, int] = {}
        # Stride-based scoring: last token count at which we triggered capture.
        self._stride_last_trigger: dict[str, int] = {}
        self.pending_hs_classification: set[str] = set()
        # On-policy mining: latest hidden state snapshot per trace (CPU list).
        self.trace_hidden_state_last: dict[str, list[float]] = {}

        # Gambit components
        self.trace_tree: TraceTree | None = None
        self.gambit_enabled = False
        # Gambit: accumulate generated token ids per request as a "full output"
        # stream (i.e., tokens generated by the model excluding the *original*
        # user prompt). For branched requests, we initialize the child's buffer
        # as a snapshot of the parent's buffer at branch time, so decoding a
        # finished branch yields an end-to-end trace including ancestors.
        #
        # We keep this buffer so we can export outputs even after finished
        # requests are removed from scheduler state.
        self._gambit_output_token_ids: dict[str, list[int]] = {}
        # Tournament mode tracking
        self._tournament_step_counter: int = 0
        self._tournament_last_round_step: int = -1
        # Sequence scorer history management: collected during tournament, flushed in batch.
        self._seq_hs_to_clear: list[str] = []
        self._seq_hs_to_copy: list[tuple[str, str]] = []

        # Async scorer: non-blocking RPC future and pending request metadata
        self._async_scorer_future = None  # Future from non_block=True collective_rpc
        self._async_scorer_meta: dict = {}  # {collect_hidden, scorer_method, req_ids}
        self._async_scorer_enabled = bool(os.environ.get("GAMBIT_ASYNC_SCORER", ""))

        # per-phase timing accumulators.
        # Set GAMBIT_PROFILE_PATH env var to enable output (e.g. "gambit_profile.jsonl").
        from collections import defaultdict as _defaultdict
        self._perf: dict[str, dict] = _defaultdict(
            lambda: {"n": 0, "total_ms": 0.0, "max_ms": 0.0}
        )
        self._perf_flush_counter: int = 0
        _profile_env = os.environ.get("GAMBIT_PROFILE_PATH", "")
        self._perf_path: str = _profile_env if _profile_env else ""

        if self.gambit_config is not None and self.gambit_config.enable_branching:
            self.gambit_enabled = True
            # Use tournament_capacity if tournament_mode, else use max_total_traces
            capacity = (
                self.gambit_config.tournament_capacity
                if self.gambit_config.tournament_mode
                else self.gambit_config.max_total_traces
            )
            self.trace_tree = TraceTree(
                max_traces=capacity,
                **self._trace_tree_branch_gates(),
            )
            if self.gambit_config.tournament_mode:
                logger.info(
                    "[Gambit] Tournament mode: capacity=%d, swap_k=%d, "
                    "check_interval=%d, warmup=%d, hard_floor=%.2f",
                    self.gambit_config.tournament_capacity,
                    self.gambit_config.tournament_swap_k,
                    self.gambit_config.tournament_check_interval,
                    self.gambit_config.tournament_warmup_tokens,
                    self.gambit_config.tournament_hard_floor,
                )
            else:
                logger.info(
                    "[Gambit] Threshold mode: max_traces=%d, theta_branch=%.2f, theta_prune=%.2f",
                    self.gambit_config.max_total_traces,
                    self.gambit_config.theta_branch,
                    self.gambit_config.theta_prune,
                )

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            ("init engine (profile, create kv cache, warmup model) took %.2f seconds"),
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)

        # Gambit: Add trace to tree if enabled
        if self.gambit_enabled and self.trace_tree is not None:
            # Initialize token accumulator for this request.
            self._gambit_output_token_ids.setdefault(request.request_id, [])
            req_id = request.request_id
            if req_id not in self.trace_tree:
                # Check if this is a child of a parallel sampling request
                # Child IDs are formatted as "{index}_{parent_id}"
                if "_" in req_id and not req_id.startswith("gambit_"):
                    # This is a parallel sampling child - add as root
                    self.trace_tree.add_root(req_id)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            # Safety: avoid hanging the frontend "drain Gambit branches" loop.
            #
            # `vllm/entrypoints/llm.py::_run_engine()` will keep calling
            # `llm_engine.step()` while `TraceTree.stats.active_count > 0`.
            # If a request disappears from the scheduler (e.g., aborted/errored)
            # but its TraceNode remains `is_active=True`, then:
            #   - scheduler.has_requests() == False (no GPU work)
            #   - TraceTree.active_count > 0 (frontend waits forever)
            #
            # In that situation, nothing can make progress anymore, so mark any
            # leftover active traces as pruned to unblock generation.
            if self.gambit_enabled and self.trace_tree is not None:
                try:
                    active_left = [
                        rid for rid, node in self.trace_tree.nodes.items()
                        if getattr(node, "is_active", False)
                    ]
                    if active_left:
                        logger.warning(
                            "[Gambit] TraceTree reports %d active traces but scheduler has no requests. "
                            "Marking them pruned to avoid hanging (ids=%s).",
                            len(active_left),
                            active_left[:8],
                        )
                        for rid in active_left:
                            self.trace_tree.mark_pruned(rid)
                        # Clear any pending stop bookkeeping for these ids.
                        self.pending_force_stops.difference_update(active_left)
                        self.pending_low_score_stops.difference_update(active_left)
                except Exception:
                    # Never raise from this safety path.
                    pass
            return {}, False
        _step_t0 = time.perf_counter()

        with record_function_or_nullcontext("core step: schedule"):
            with self._timed_phase("schedule"):
                scheduler_output = self.scheduler.schedule()

        with record_function_or_nullcontext("core step: execute_model"):
            with self._timed_phase("model_execute"):
                future = self.model_executor.execute_model(scheduler_output, non_block=True) # forward
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                with self.log_error_detail(scheduler_output):
                    model_output = future.result()
                    if model_output is None:
                        model_output = self.model_executor.sample_tokens(grammar_output) # sample and process logit
        
        if self.gambit_config.enable:
            # 1) For requests marked in the previous step, read/score after the current forward completes.
            # --- Async scorer: consume results from PREVIOUS step's non-blocking RPC ---
            if self._async_scorer_enabled and self._async_scorer_future is not None:
                with self._timed_phase("scorer_rpc"):
                    scores_per_worker = self._async_scorer_future.result()
                    self._async_scorer_future = None
                collect_hidden = self._async_scorer_meta.get("collect_hidden", False)
                # Process scores (same logic as sync path)
                scores: dict[str, float | list[float]] = {}
                for worker_payload in scores_per_worker:
                    for rid, payload in worker_payload.items():
                        if collect_hidden and isinstance(payload, dict):
                            scores[rid] = payload.get("scores", [])
                            hs = payload.get("hidden_state")
                            if hs is not None:
                                self.trace_hidden_state_last[rid] = hs
                        else:
                            scores[rid] = payload
                for rid, score in scores.items():
                    if (
                        self.gambit_config is not None
                        and self.gambit_config.record_score_history
                    ):
                        buf = self.trace_scores.setdefault(rid, [])
                        if isinstance(score, (list, tuple)):
                            buf.extend(score)
                        else:
                            buf.append(score)
                    else:
                        if isinstance(score, (list, tuple)):
                            s = float(sum(score)) if score else 0.0
                            c = int(len(score))
                        else:
                            s = float(score)
                            c = 1
                        if c > 0:
                            self.trace_score_sums[rid] = (
                                float(self.trace_score_sums.get(rid, 0.0)) + s
                            )
                            self.trace_score_counts[rid] = (
                                int(self.trace_score_counts.get(rid, 0)) + c
                            )
                if scores:
                    prev_pending = self._async_scorer_meta.get("req_ids", set())
                    self.pending_hs_classification.difference_update(prev_pending)
                with self._timed_phase("score_aggregation"):
                    if (
                        self.gambit_config is not None
                        and self.gambit_config.record_score_history
                    ):
                        trace_avg_scores = {
                            rid: (sum(buf) / len(buf))
                            for rid, buf in self.trace_scores.items()
                            if buf
                        }
                    else:
                        trace_avg_scores = {
                            rid: (self.trace_score_sums[rid] / self.trace_score_counts[rid])
                            for rid in self.trace_score_counts
                            if self.trace_score_counts.get(rid, 0) > 0
                        }
                    if trace_avg_scores:
                        self.scheduler.update_trace_scores(trace_avg_scores)
                if self.gambit_enabled and trace_avg_scores:
                    with self._timed_phase("decision_step_total"):
                        self._gambit_decision_step(trace_avg_scores)

            if self.pending_hs_classification:
                num_scheduled_tokens = {
                    rid: scheduler_output.num_scheduled_tokens[rid]
                    for rid in self.pending_hs_classification
                    if rid in scheduler_output.num_scheduled_tokens
                }
                if num_scheduled_tokens:
                    collect_hidden = bool(
                        self.gambit_config is not None
                        and self.gambit_config.collect_hidden_states_for_mining
                    )
                    scorer_method = (
                        "step_scorer_evaluate_with_hidden_states"
                        if collect_hidden else "step_scorer_evaluate"
                    )
                    if self._async_scorer_enabled:
                        # --- Async scorer: submit non-blocking RPC, consume next step ---
                        with self._timed_phase("scorer_rpc"):
                            self._async_scorer_future = self.collective_rpc(
                                scorer_method,
                                args=(list(num_scheduled_tokens.keys()), num_scheduled_tokens),
                                non_block=True,
                            )
                            self._async_scorer_meta = {
                                "collect_hidden": collect_hidden,
                                "req_ids": set(num_scheduled_tokens.keys()),
                            }
                        # Skip inline score processing — will be consumed next step
                        scores: dict[str, float | list[float]] = {}
                        trace_avg_scores: dict[str, float] = {}
                    else:
                        # --- Sync scorer (original path) ---
                        # scorer RPC blocks CPU until GPU worker returns scores.
                        with self._timed_phase("scorer_rpc"):
                            scores_per_worker = self.collective_rpc(
                                scorer_method,
                                args=(list(num_scheduled_tokens.keys()), num_scheduled_tokens),
                            )

                    if not self._async_scorer_enabled:
                        scores: dict[str, float | list[float]] = {}
                        for worker_payload in scores_per_worker:
                            for rid, payload in worker_payload.items():
                                if collect_hidden and isinstance(payload, dict):
                                    scores[rid] = payload.get("scores", [])
                                    hs = payload.get("hidden_state")
                                    if hs is not None:
                                        self.trace_hidden_state_last[rid] = hs
                                else:
                                    scores[rid] = payload

                        for rid, score in scores.items():
                            if (
                                self.gambit_config is not None
                                and self.gambit_config.record_score_history
                            ):
                                buf = self.trace_scores.setdefault(rid, [])
                                if isinstance(score, (list, tuple)):
                                    buf.extend(score)
                                else:
                                    buf.append(score)
                            else:
                                if isinstance(score, (list, tuple)):
                                    s = float(sum(score)) if score else 0.0
                                    c = int(len(score))
                                else:
                                    s = float(score)
                                    c = 1
                                if c > 0:
                                    self.trace_score_sums[rid] = (
                                        float(self.trace_score_sums.get(rid, 0.0)) + s
                                    )
                                    self.trace_score_counts[rid] = (
                                        int(self.trace_score_counts.get(rid, 0)) + c
                                    )

                        if scores:
                            self.pending_hs_classification.difference_update(scores.keys())

                        with self._timed_phase("score_aggregation"):
                            if (
                                self.gambit_config is not None
                                and self.gambit_config.record_score_history
                            ):
                                trace_avg_scores = {
                                    rid: (sum(buf) / len(buf))
                                    for rid, buf in self.trace_scores.items()
                                    if buf
                                }
                            else:
                                trace_avg_scores = {
                                    rid: (self.trace_score_sums[rid] / self.trace_score_counts[rid])
                                    for rid in self.trace_score_counts
                                    if self.trace_score_counts.get(rid, 0) > 0
                                }
                            if trace_avg_scores:
                                self.scheduler.update_trace_scores(trace_avg_scores)

                        # Gambit: Make branching and pruning decisions based on scores
                        if self.gambit_enabled and trace_avg_scores:
                            with self._timed_phase("decision_step_total"):
                                self._gambit_decision_step(trace_avg_scores)

            # Finished requests no longer need to wait for classification.
            if self.scheduler.finished_req_ids:
                self.pending_hs_classification.difference_update(
                    self.scheduler.finished_req_ids
                )

            # 2) if a step boundary is detected, mark the req for classification in the next step.
            trigger_req_ids: list[str] = []
            scoring_stride = self.gambit_config.scoring_stride
            for rid, idx in model_output.req_id_to_index.items():
                sampled = model_output.sampled_token_ids[idx]
                total_tokens = self.trace_token_counts.get(rid, 0) + len(sampled)
                self.trace_token_counts[rid] = total_tokens

                triggered = False
                if scoring_stride > 0:
                    # Stride-based trigger: fire when token count crosses
                    # the next stride boundary.
                    last_trigger = self._stride_last_trigger.get(rid, 0)
                    if total_tokens - last_trigger >= scoring_stride:
                        triggered = True
                        self._stride_last_trigger[rid] = total_tokens
                else:
                    # Default: trigger on \n\n tokens.
                    boundary_ids = self.gambit_config.double_new_line_tokenID
                    if boundary_ids and any(
                        token_id in boundary_ids for token_id in sampled
                    ):
                        triggered = True

                if triggered:
                    trigger_req_ids.append(rid)

            if trigger_req_ids:
                self.pending_hs_classification.update(trigger_req_ids)

            # Gambit: Handle proactive pruning by inserting EOS tokens
            if self.gambit_enabled:
                with self._timed_phase("eos_injection"):
                    if self.pending_low_score_stops:
                        prune_req_ids = [
                            rid for rid in self.pending_low_score_stops
                            if rid in model_output.req_id_to_index
                        ]
                        if prune_req_ids:
                            self._insert_eos(
                                model_output, prune_req_ids, stop_reason="gambit_pruned"
                            )
                            # Clear the pruned requests from pending
                            self.pending_low_score_stops.difference_update(prune_req_ids)

                    if self.pending_force_stops:
                        force_req_ids = [
                            rid for rid in self.pending_force_stops
                            if rid in model_output.req_id_to_index
                        ]
                        if force_req_ids:
                            self._insert_eos(
                                model_output,
                                force_req_ids,
                                stop_reason="gambit_stop_after_completed",
                            )
                            self.pending_force_stops.difference_update(force_req_ids)

        # Gambit: capture generated tokens for export/analysis.
        # Do this after any EOS insertion so the captured stream reflects pruning.
        if self.gambit_enabled:
            for rid, idx in model_output.req_id_to_index.items():
                sampled = model_output.sampled_token_ids[idx]
                if not sampled:
                    continue
                buf = self._gambit_output_token_ids.setdefault(rid, [])
                # sampled is typically list[int]; be defensive.
                buf.extend(int(t) for t in sampled)

        # Prepare score histories for finished requests before update_from_output
        # We need to capture them now because _finalize_scores will clean them up
        trace_score_histories_for_output = None
        if (
            self.gambit_config is not None
            and self.gambit_config.enable
            and self.gambit_config.record_score_history
        ):
            trace_score_histories_for_output = {
                rid: list(scores) for rid, scores in self.trace_scores.items()
            }

        with record_function_or_nullcontext("core step: update_from_output"):
            with self._timed_phase("output_process"):
                engine_core_outputs = self.scheduler.update_from_output(  # handle eos
                    scheduler_output,
                    model_output,
                    final_trace_scores=self.final_trace_scores,
                    trace_score_histories=trace_score_histories_for_output,
                )
        if self.scheduler.finished_req_ids:
            self._finalize_scores(self.scheduler.finished_req_ids)
        # Attach final trace scores to finished outputs for front-end consumption.
        if self.final_trace_scores:
            for eco in engine_core_outputs.values():
                for output in eco.outputs:
                    if output.finish_reason is not None and output.final_score is None:
                        output.final_score = self.final_trace_scores.get(
                            output.request_id
                        )

        # Gambit: Attach tree export to outputs when requests finish
        if self.gambit_enabled and self.scheduler.finished_req_ids:
            gambit_export = self.get_gambit_tree_export()
            if gambit_export is not None:
                for eco in engine_core_outputs.values():
                    eco.gambit_tree_export = gambit_export

        # Record end-to-end step time
        _step_dur_ms = (time.perf_counter() - _step_t0) * 1000.0
        p = self._perf["step_total"]
        p["n"] += 1
        p["total_ms"] += _step_dur_ms
        if _step_dur_ms > p["max_ms"]:
            p["max_ms"] = _step_dur_ms

        # Periodic profiling flush (every 200 steps)
        if self._perf_path and p["n"] % 200 == 0:
            self._perf_flush()

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _finalize_scores(self, finished_ids: Sequence[str]) -> None:
        for rid in finished_ids:
            if (
                self.gambit_config is not None
                and self.gambit_config.enable
                and not self.gambit_config.record_score_history
            ):
                c = int(self.trace_score_counts.get(rid, 0) or 0)
                s = float(self.trace_score_sums.get(rid, 0.0) or 0.0)
                final_avg = (s / c) if c > 0 else None
            else:
                scores = self.trace_scores.get(rid, [])
                final_avg = sum(scores) / len(scores) if scores else None
            self.final_trace_scores[rid] = final_avg
            self.trace_scores.pop(rid, None)
            self.trace_score_sums.pop(rid, None)
            self.trace_score_counts.pop(rid, None)
            self.pending_low_score_stops.discard(rid)
            self.pending_force_stops.discard(rid)
            self.trace_token_counts.pop(rid, None)
            self._stride_last_trigger.pop(rid, None)

            # Gambit: Mark trace as completed in the tree with output data
            if self.gambit_enabled:
                output_token_ids = self._gambit_output_token_ids.pop(rid, None)
                self._gambit_on_request_finished(
                    rid, final_avg, output_token_ids=output_token_ids
                )

    
    def _insert_eos(self, model_output, request_idx, stop_reason: str | None = None):
        # Must be the same value `check_stop` compares against (Request.eos_token_id,
        # which comes from the tokenizer). hf_config.eos_token_id is a different
        # field and is a *list* for several model families, in which case the
        # forced stop would never match and the pruned trace would keep decoding.
        if stop_reason:
            self.scheduler.mark_stop_reason(request_idx, stop_reason)
        for i in request_idx:
            if i not in model_output.req_id_to_index:
                continue
            request = self.scheduler.requests.get(i)
            eos_token_id = getattr(request, "eos_token_id", None)
            if not isinstance(eos_token_id, int):
                logger.warning(
                    "Cannot force-stop trace %s: no scalar EOS token id available.", i
                )
                continue
            model_output.sampled_token_ids[model_output.req_id_to_index[i]] = [eos_token_id]
        return

    @contextmanager
    def _timed_phase(self, phase: str):
        # measure wall-clock duration of a code region.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            p = self._perf[phase]
            p["n"] += 1
            p["total_ms"] += dur_ms
            if dur_ms > p["max_ms"]:
                p["max_ms"] = dur_ms

    def _perf_flush(self) -> None:
        # append a cumulative snapshot to the JSONL file.
        if not self._perf_path:
            return
        snapshot = {
            "flush_idx": self._perf_flush_counter,
            "tournament_step": self._tournament_step_counter,
            "phases": {k: dict(v) for k, v in self._perf.items()},
        }
        self._perf_flush_counter += 1
        try:
            with open(self._perf_path, "a") as _pf:
                _pf.write(json.dumps(snapshot) + "\n")
        except Exception:
            pass

    # Gambit Methods

    def _gambit_decision_step(self, trace_avg_scores: dict[str, float]) -> None:
        """
        Make Gambit branching and pruning decisions based on current scores.

        Routes to either tournament mode or threshold mode based on config.
        """
        if not self.gambit_enabled or self.trace_tree is None:
            return

        # Increment step counter
        self.trace_tree.increment_step()
        self._tournament_step_counter += 1

        # Ensure all traces are in the tree and record scores
        for req_id, avg_score in trace_avg_scores.items():
            # Recording full per-step score trajectories can be extremely memory
            # intensive for long generations. If disabled, we still use
            # `trace_avg_scores` for tournament/threshold decisions, but avoid
            # appending to TraceTree nodes.
            # The node must exist first: record_score is a no-op on an unknown
            # req_id, which would silently drop each trace's first score.
            if req_id not in self.trace_tree:
                self.trace_tree.add_root(req_id)
            if (
                self.gambit_config is None
                or self.gambit_config.record_score_history
            ):
                self.trace_tree.record_score(req_id, avg_score)

        # Optional global stop condition: once we have "enough" completed traces
        # for this question, forcibly stop the remaining active traces to avoid
        # wasting tokens on long continuations.
        if self._gambit_should_stop_after_completed():
            self._gambit_force_stop_all_active_traces()
            return

        # Route to appropriate mode
        if self.gambit_config is not None and self.gambit_config.tournament_mode:
            # tournament step: sort + prune + branch logic.
            with self._timed_phase("tournament_step"):
                self._gambit_tournament_step(trace_avg_scores)
        else:
            # threshold step: fixed-threshold prune/branch (no tournament).
            with self._timed_phase("threshold_step"):
                self._gambit_threshold_step(trace_avg_scores)

        # Flush sequence-scorer history sync (prune clears, branch copies) in one batch RPC.
        if self._seq_hs_to_clear or self._seq_hs_to_copy:
            # sync_sequence_histories RPC: blocks until worker
            # copies/clears sequence history tensors for all pruned/branched requests.
            with self._timed_phase("sync_sequence_rpc"):
                self.collective_rpc(
                    "sync_sequence_histories",
                    args=(list(self._seq_hs_to_clear), list(self._seq_hs_to_copy)),
                )
            self._seq_hs_to_clear.clear()
            self._seq_hs_to_copy.clear()

        # Log stats periodically; flush profiling snapshot at same interval.
        log_interval = 5 if self.trace_tree.current_step < 20 else 50
        if self.trace_tree.current_step % log_interval == 0:
            self._gambit_log_stats(trace_avg_scores)
            # write snapshot to disk at log interval.
            self._perf_flush()

    def _gambit_tournament_step(self, trace_avg_scores: dict[str, float]) -> None:
        """
        Tournament-style selection: rank-based branching and pruning.

        Algorithm:
        1. Hard floor: Kill any trace with score < hard_floor (garbage collection)
        2. Check if it's time for a tournament round (every check_interval steps)
        3. If N < C: Branch top (C-N) traces to fill capacity
        4. If N == C: Swap bottom K traces with clones of top K traces
        """
        if self.gambit_config is None or self.trace_tree is None:
            return

        config = self.gambit_config
        capacity = config.tournament_capacity
        swap_k = config.tournament_swap_k
        check_interval = config.tournament_check_interval
        warmup_tokens = config.tournament_warmup_tokens
        hard_floor = config.tournament_hard_floor

        # Step A: Hard floor - garbage collect any trace with score < hard_floor
        # hard_floor_prune: identifies and prunes sub-threshold traces.
        with self._timed_phase("hard_floor_prune"):
            traces_to_prune = []
            for req_id, avg_score in trace_avg_scores.items():
                if req_id in self.pending_low_score_stops:
                    continue
                num_tokens = self.trace_token_counts.get(req_id, 0)
                if num_tokens >= warmup_tokens and avg_score < hard_floor:
                    traces_to_prune.append(req_id)

            for req_id in traces_to_prune:
                self._gambit_prune_trace(req_id)
                logger.debug(
                    "[Tournament] Hard floor prune: %s score=%.3f < %.3f",
                    req_id, trace_avg_scores.get(req_id, 0), hard_floor
                )

        # Check if we should run a tournament round
        steps_since_last = self._tournament_step_counter - self._tournament_last_round_step
        if self._tournament_last_round_step < 0:
            # First round - check if any trace has enough tokens
            any_ready = any(
                self.trace_token_counts.get(rid, 0) >= warmup_tokens
                for rid in trace_avg_scores.keys()
            )
            if not any_ready:
                return
        elif steps_since_last < check_interval:
            return

        # Time for a tournament round!
        self._tournament_last_round_step = self._tournament_step_counter

        # Get scored, active traces with their scores.
        #
        # `warmup_tokens` gates when tournament behaviour starts and avoids
        # hard-floor pruning on early, noisy scores. Capacity management must
        # still use the true active count rather than the warmup-ready count,
        # otherwise the pool stays in "fill slot" mode and never reaches the
        # swap/prune path.
        active_traces: list[tuple[float, str]] = []
        for req_id, avg_score in trace_avg_scores.items():
            if req_id in self.pending_low_score_stops:
                continue
            node = self.trace_tree.get_node(req_id)
            if node is None or not node.is_active:
                continue
            active_traces.append((avg_score, req_id))

        if not active_traces:
            return

        # Sort by score (highest first)
        # tournament_sort: O(N log N) sort of all active traces;
        # runs every check_interval steps and is the primary CPU cost of each tournament round.
        with self._timed_phase("tournament_sort"):
            active_traces.sort(key=lambda x: -x[0])
        scored_n = len(active_traces)
        active_n = self.trace_tree.get_active_count()

        # To prevent runaway "branch-of-branch-of-branch" behavior, only allow a
        # trace to be selected as a *parent* for branching once it has generated
        # enough tokens since its own creation (warmup gate). Newly spawned
        # branches start with `trace_token_counts[child] = 0`, so they won't be
        # eligible until they generate `warmup_tokens`.
        branch_candidates: list[tuple[float, str]] = [
            (score, rid)
            for (score, rid) in active_traces
            if self.trace_token_counts.get(rid, 0) >= warmup_tokens
        ]

        logger.info(
            "[Tournament] Round at step %d: active=%d, scored=%d, C=%d, scores=[%.3f-%.3f]",
            self._tournament_step_counter,
            active_n,
            scored_n,
            capacity,
            active_traces[-1][0] if active_traces else 0,
            active_traces[0][0] if active_traces else 0,
        )

        # Step B: Rank and Balance
        # tournament_swap: prune bottom-K + branch top-K;
        # each _gambit_create_branch call builds a new request and copies score history.
        with self._timed_phase("tournament_swap"):
            if active_n < capacity:
                # We have empty slots - branch top (C-active) traces.
                # Ablation: if tournament_branch_random, sample uniformly
                # among eligible branch_candidates instead of taking top-K.
                slots_available = capacity - active_n
                if self.gambit_config.tournament_branch_random and branch_candidates:
                    n_sel = min(slots_available, len(branch_candidates))
                    traces_to_branch = random.sample(branch_candidates, n_sel)
                else:
                    traces_to_branch = branch_candidates[:slots_available]  # Top eligible traces

                for score, req_id in traces_to_branch:
                    # Double-check capacity before each branch
                    if self.trace_tree.get_active_count() >= capacity:
                        break
                    if self._gambit_create_branch(req_id):
                        logger.debug(
                            "[Tournament] Branch (fill slot): %s score=%.3f",
                            req_id, score
                        )
            elif active_n >= capacity:
                # At or over capacity - swap bottom K with clones of top K
                # First, ensure we don't exceed capacity by pruning excess
                # active_n counts every logically-active node (including ghosts),
                # while active_traces holds only the scored ones. Clamp so an
                # excess larger than the scored pool cannot prune all of it.
                excess = min(active_n - capacity, len(active_traces))
                if excess > 0:
                    # Prune the excess lowest-scoring traces
                    bottom_excess = active_traces[-(excess):]
                    for score, req_id in bottom_excess:
                        self._gambit_prune_trace(req_id)
                        logger.debug(
                            "[Tournament] Prune (over capacity): %s score=%.3f",
                            req_id, score
                        )
                    # Update active_traces list
                    active_traces = active_traces[:-excess]
                    scored_n = len(active_traces)

                # Now swap: prune bottom K (by score), branch top K (or random K).
                # Ablation: when tournament_branch_random, BRANCH selection is
                # uniform-random over eligible parents; PRUNE selection stays
                # bottom-K by score (we're ablating score-guided branching, not
                # score-guided pruning).
                actual_k = min(swap_k, scored_n // 2)  # Don't swap more than half of scored
                if actual_k > 0:
                    bottom_k = active_traces[-actual_k:]  # Lowest scores
                    # A trace can be both bottom-K by score and warmup-eligible.
                    # Pruning runs first, so leaving it in the parent pool would
                    # silently drop the branch; exclude it up front.
                    victim_ids = {req_id for _, req_id in bottom_k}
                    eligible_parents = [
                        (score, req_id)
                        for score, req_id in branch_candidates
                        if req_id not in victim_ids
                    ]
                    if self.gambit_config.tournament_branch_random and eligible_parents:
                        k_branch = min(actual_k, len(eligible_parents))
                        top_k = random.sample(eligible_parents, k_branch)
                    else:
                        top_k = eligible_parents[:actual_k]  # Highest eligible parents

                    # Prune bottom K
                    for score, req_id in bottom_k:
                        self._gambit_prune_trace(req_id)
                        logger.debug(
                            "[Tournament] Prune (swap out): %s score=%.3f",
                            req_id, score
                        )

                    # Branch top K (checking capacity each time)
                    for score, req_id in top_k:
                        if self.trace_tree.get_active_count() >= capacity:
                            break
                        if self._gambit_create_branch(req_id):
                            logger.debug(
                                "[Tournament] Branch (swap in clone): %s score=%.3f",
                                req_id, score
                            )

    def _gambit_threshold_step(self, trace_avg_scores: dict[str, float]) -> None:
        """
        Legacy threshold-based selection.

        Prunes traces below theta_prune, branches traces above theta_branch,
        enforcing the capacity limit.
        """
        if self.gambit_config is None or self.trace_tree is None:
            return

        traces_to_prune: list[str] = []
        traces_to_branch: list[str] = []

        for req_id, avg_score in trace_avg_scores.items():
            num_tokens = self.trace_token_counts.get(req_id, 0)

            # Check for proactive pruning (score too low)
            if self._gambit_should_prune_threshold(req_id, avg_score, num_tokens):
                traces_to_prune.append(req_id)
                continue

            # Check for branching (score high enough)
            # Note: capacity check is done during execution, not here
            if self._gambit_should_branch_threshold(req_id, avg_score, num_tokens):
                traces_to_branch.append(req_id)

        # Execute pruning first (this frees up capacity)
        for req_id in traces_to_prune:
            self._gambit_prune_trace(req_id)

        # Sort branches by score (highest first) and execute with capacity check
        branch_scores = [(trace_avg_scores.get(rid, 0), rid) for rid in traces_to_branch]
        branch_scores.sort(key=lambda x: -x[0])

        for score, req_id in branch_scores:
            # Check capacity before each branch creation.
            if self.trace_tree.get_active_count() >= self.trace_tree.max_traces:
                logger.debug(
                    "[Gambit] Skipping branch for %s: at capacity (%d/%d)",
                    req_id, self.trace_tree.get_active_count(), self.trace_tree.max_traces
                )
                break
            self._gambit_create_branch(req_id)

    def _gambit_should_prune_threshold(
        self, req_id: str, avg_score: float, num_tokens: int
    ) -> bool:
        """Check if a trace should be proactively pruned (threshold mode)."""
        if self.gambit_config is None:
            return False

        # Don't prune before minimum tokens
        if num_tokens < self.gambit_config.min_tokens_before_prune:
            return False

        # Don't prune if already marked for pruning
        if req_id in self.pending_low_score_stops:
            return False

        # Prune if score is below threshold
        return avg_score < self.gambit_config.theta_prune

    def _gambit_should_branch_threshold(
        self, req_id: str, avg_score: float, num_tokens: int
    ) -> bool:
        """Check if a trace should branch (threshold mode, without capacity check)."""
        if self.gambit_config is None or self.trace_tree is None:
            return False

        # Check score threshold
        if avg_score <= self.gambit_config.theta_branch:
            return False

        # Check minimum tokens
        if num_tokens < self.gambit_config.min_tokens_before_branch:
            return False

        # Check cooldown
        node = self.trace_tree.get_node(req_id)
        if node is None or not node.is_active:
            return False

        if node.last_branch_step >= 0:
            steps_since = self.trace_tree.current_step - node.last_branch_step
            if steps_since < self.gambit_config.branch_cooldown_steps:
                return False

        return True

    def _gambit_log_stats(self, trace_avg_scores: dict[str, float]) -> None:
        """Log Gambit statistics."""
        if self.trace_tree is None or self.gambit_config is None:
            return

        stats = self.trace_tree.get_stats()
        all_scores = list(trace_avg_scores.values())
        max_score = max(all_scores) if all_scores else 0
        min_score = min(all_scores) if all_scores else 0

        mode = "tournament" if self.gambit_config.tournament_mode else "threshold"
        capacity = (
            self.gambit_config.tournament_capacity
            if self.gambit_config.tournament_mode
            else self.gambit_config.max_total_traces
        )
        logger.info(
            "[Gambit-%s] Step %d: active=%d/%d, completed=%d, pruned=%d, branches=%d, "
            "scores=[%.3f-%.3f]",
            mode,
            stats["current_step"],
            stats["active_count"],
            capacity,
            stats["completed_count"],
            stats["pruned_count"],
            stats["branch_counter"],
            min_score,
            max_score,
        )

    def _gambit_prune_trace(self, req_id: str) -> None:
        """Proactively prune a low-scoring trace."""
        if self.trace_tree is None:
            return

        # Mark in pending stops so scheduler will terminate it
        self.pending_low_score_stops.add(req_id)
        self.trace_tree.mark_pruned(req_id)
        self._seq_hs_to_clear.append(req_id)

        logger.debug(
            "[Gambit] Proactively pruning trace %s (score=%.3f, tokens=%d)",
            req_id,
            self.trace_tree.get_score(req_id) or 0.0,
            self.trace_token_counts.get(req_id, 0),
        )

    def _gambit_should_stop_after_completed(self) -> bool:
        """Return True if we should stop remaining traces due to completion limit."""
        if self.trace_tree is None or self.gambit_config is None:
            return False
        limit = int(getattr(self.gambit_config, "stop_after_completed_traces", 0) or 0)
        if limit <= 0:
            return False
        return self.trace_tree.get_completed_count() >= limit

    def _gambit_force_stop_all_active_traces(self) -> None:
        """Force-stop all active traces (used when completion limit is reached)."""
        if self.trace_tree is None or self.gambit_config is None:
            return
        limit = int(getattr(self.gambit_config, "stop_after_completed_traces", 0) or 0)
        completed = self.trace_tree.get_completed_count()
        active_ids = [
            rid for rid, node in self.trace_tree.nodes.items()
            if node.is_active and rid not in self.pending_force_stops
        ]
        if not active_ids:
            return
        logger.info(
            "[Gambit] stop_after_completed_traces reached (%d/%d). "
            "Force-stopping %d active traces.",
            completed,
            limit,
            len(active_ids),
        )
        for rid in active_ids:
            # Mark as pruned so it won't be treated as a completed trace.
            self.trace_tree.mark_pruned(rid)
            self.pending_force_stops.add(rid)
        logger.debug(
            "After force-stop mark_pruned: tree.active_count=%d, "
            "pending_force_stops=%d, scheduler.running=%d, scheduler.waiting=%d",
            self.trace_tree.get_active_count(),
            len(self.pending_force_stops),
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )

    def _gambit_create_branch(self, parent_req_id: str) -> bool:
        """Create a new branch from a high-scoring trace."""
        if self.trace_tree is None or self.gambit_config is None:
            return False

        # If we've already collected enough completed traces for this question,
        # don't spawn any new branches.
        if self._gambit_should_stop_after_completed():
            return False

        # Enforce per-trace branching constraints (budget + min tokens + cooldown).
        # `trace_token_counts` is "tokens generated since branch point" (children
        # start at 0), which prevents immediate re-branching of newly spawned
        # children and limits branching frequency.
        parent_new_tokens = int(self.trace_token_counts.get(parent_req_id, 0) or 0)
        if not self.trace_tree.can_trace_branch(parent_req_id, parent_new_tokens):
            return False

        # Get parent request from scheduler
        parent_request = self.scheduler.requests.get(parent_req_id)
        if parent_request is None:
            logger.warning("[Gambit] Parent request not found: %s", parent_req_id)
            return False

        # Generate unique child request ID
        child_req_id = self.trace_tree.generate_branch_id(parent_req_id)

        # Create branched trace in the tree. Record the inherited prefix length
        # so downstream code can compute token counts "since branch" without
        # double-counting ancestor tokens.
        parent_output_len = len(self._gambit_output_token_ids.get(parent_req_id, []))
        child_node = self.trace_tree.add_branch(
            parent_req_id,
            child_req_id,
            branch_parent_output_len=parent_output_len,
        )
        if child_node is None:
            return False

        # Create the child request with same prefix (leverages vLLM prefix caching)
        # The child request uses all tokens generated so far as its "prompt"
        with self._timed_phase("branch_request_create"):
            child_request = self._gambit_create_child_request(parent_request, child_req_id)
        if child_request is None:
            self.trace_tree.remove_trace(child_req_id)
            return False

        # Add child request to scheduler (triggers KV cache prefix matching)
        with self._timed_phase("branch_scheduler_add"):
            self.scheduler.add_request(child_request)
        self._seq_hs_to_copy.append((parent_req_id, child_req_id))

        # Copy score tracking state
        if (
            self.gambit_config is not None
            and self.gambit_config.enable
            and not self.gambit_config.record_score_history
        ):
            if parent_req_id in self.trace_score_counts:
                self.trace_score_counts[child_req_id] = int(
                    self.trace_score_counts.get(parent_req_id, 0) or 0
                )
                self.trace_score_sums[child_req_id] = float(
                    self.trace_score_sums.get(parent_req_id, 0.0) or 0.0
                )
        else:
            parent_scores = self.trace_scores.get(parent_req_id, [])
            if parent_scores:
                self.trace_scores[child_req_id] = parent_scores.copy()

        self.trace_token_counts[child_req_id] = 0  # Child starts with 0 new tokens
        self._stride_last_trigger[child_req_id] = 0
        # Gambit: Child's "full output" buffer should include the parent's output
        # tokens up to the moment of branching, so branch decoding is end-to-end.
        self._gambit_output_token_ids[child_req_id] = list(
            self._gambit_output_token_ids.get(parent_req_id, [])
        )

        logger.debug(
            "[Gambit] Created branch %s from %s (parent_score=%.3f, depth=%d)",
            child_req_id,
            parent_req_id,
            self.trace_tree.get_score(parent_req_id) or 0.0,
            child_node.branch_depth,
        )
        return True

    def _gambit_create_child_request(
        self, parent_request: Request, child_req_id: str
    ) -> Request | None:
        """
        Create a child request that branches from the parent.

        The child request uses all of the parent's tokens (prompt + output)
        as its prefix, allowing vLLM's prefix caching to reuse the KV cache.
        """
        if self.gambit_config is None:
            return None

        # Get all tokens from parent (prompt + generated output)
        prefix_tokens = list(parent_request.all_token_ids)

        # Create varied sampling params for diversity
        child_sampling_params = None
        if parent_request.sampling_params is not None:
            child_sampling_params = copy(parent_request.sampling_params)
            # Boost temperature slightly for diversity
            if child_sampling_params.temperature is not None:
                child_sampling_params.temperature *= self.gambit_config.branch_temperature_boost
            # Use different seed for different sampling path
            if child_sampling_params.seed is not None:
                child_sampling_params.seed = (
                    child_sampling_params.seed + self.trace_tree.branch_counter
                    if self.trace_tree else child_sampling_params.seed + 1
                )
            else:
                # Generate a new seed based on branch counter
                child_sampling_params.seed = random.randint(0, 2**31 - 1)

        try:
            child_request = Request(
                request_id=child_req_id,
                prompt_token_ids=prefix_tokens,
                sampling_params=child_sampling_params,
                pooling_params=parent_request.pooling_params,
                eos_token_id=parent_request.eos_token_id,
                client_index=parent_request.client_index,
                arrival_time=parent_request.arrival_time,
                lora_request=parent_request.lora_request,
                cache_salt=parent_request.cache_salt,
                priority=parent_request.priority,
                trace_headers=parent_request.trace_headers,
                block_hasher=self.request_block_hasher,
            )
            return child_request
        except Exception as e:
            logger.error("[Gambit] Failed to create child request: %s", e)
            return None

    def _gambit_on_request_finished(
        self,
        req_id: str,
        final_score: float | None = None,
        *,
        output_token_ids: list[int] | None = None,
    ) -> None:
        """Called when a request finishes (completed or pruned)."""
        if self.trace_tree is None:
            return

        node = self.trace_tree.get_node(req_id)
        if node is None:
            # Try to add as root if not already in tree
            if req_id not in self.trace_tree:
                self.trace_tree.add_root(req_id)
                node = self.trace_tree.get_node(req_id)
            if node is None:
                return

        # Capture output data (even for pruned traces).
        output_text = None
        output_token_ids_local = output_token_ids
        finish_reason = None
        stop_reason = None

        # Try to get request from scheduler to capture output.
        # Note: finished requests may already have been removed; in that case we
        # rely on the `output_token_ids` buffer passed in from `_finalize_scores`.
        request = self.scheduler.requests.get(req_id)
        if request is not None:
            # Scheduler holds the authoritative finish/stop metadata.
            try:
                output_token_ids_local = list(request.output_token_ids)
            except Exception:
                pass
            finish_reason = request.status.name if request.status else None
            stop_reason = request.stop_reason

        if node.is_pruned:
            # Do NOT mark as completed (we don't want pruned traces counted as
            # "completed"), but do record partial outputs for analysis.
            node.set_output(
                output_text=output_text,
                output_token_ids=output_token_ids_local,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                final_score=final_score,
            )
        else:
            self.trace_tree.mark_completed(
                req_id,
                output_text=output_text,
                output_token_ids=output_token_ids_local,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                final_score=final_score,
            )

    def get_gambit_tree_export(self) -> dict | None:
        """
        Export the Gambit trace tree structure for external use.

        Returns a dictionary with tree stats, nodes, and roots,
        or None if Gambit is not enabled.
        """
        if not self.gambit_enabled or self.trace_tree is None:
            return None
        export = self.trace_tree.export_tree()
        stats = export.get("stats") or {}
        logger.debug(
            "get_gambit_tree_export: active=%d completed=%d pruned=%d "
            "scheduler.running=%d pending_force_stops=%d",
            stats.get("active_count", -1),
            stats.get("completed_count", -1),
            stats.get("pruned_count", -1),
            len(self.scheduler.running),
            len(self.pending_force_stops),
        )
        if self.gambit_config is not None and self.gambit_config.collect_hidden_states_for_mining:
            nodes = export.get("nodes")
            if isinstance(nodes, dict):
                for rid, node_data in nodes.items():
                    if rid in self.trace_hidden_state_last and isinstance(node_data, dict):
                        node_data["hidden_state"] = self.trace_hidden_state_last[rid]
        # inject scheduler_active_count so the drain loop in
        # llm.py can exit on a ghost-free guard.  Ghost traces remain is_active=True
        # after _stop_lowest_running() removes them from scheduler.requests, so
        # active_count never reaches 0 and the drain loop deadlocks forever.
        # len(waiting) + len(running) counts only real in-flight branches and hits
        # 0 exactly when the last branch completes — the correct exit point.
        if isinstance(export.get("stats"), dict):
            export["stats"]["scheduler_active_count"] = (
                len(self.scheduler.waiting) + len(self.scheduler.running)
            )
        return export

    def _trace_tree_branch_gates(self) -> dict[str, int]:
        """Per-trace branch gates for the TraceTree.

        In tournament mode the warmup threshold `w` is the only eligibility
        rule the algorithm defines, so the legacy threshold-mode cooldown and
        minimum-token gates are disabled. Leaving them on would silently
        suppress branches whenever the check interval is shorter than
        `branch_cooldown_steps` (both are counted in decision rounds).
        """
        if self.gambit_config.tournament_mode:
            return {"branch_cooldown_steps": 0, "min_tokens_before_branch": 0}
        return {
            "branch_cooldown_steps": self.gambit_config.branch_cooldown_steps,
            "min_tokens_before_branch": self.gambit_config.min_tokens_before_branch,
        }

    def reset_gambit_state(self) -> None:
        """
        Reset Gambit bookkeeping state (trace tree + token buffers) to avoid
        leaking traces across independent generate() calls.

        Safe to call when there are no active scheduler requests.
        """
        if not self.gambit_enabled or self.gambit_config is None:
            return

        # Guard against resetting while the scheduler still has real in-flight
        # requests.  Use the scheduler's own queues (not trace_tree.get_active_count())
        # because ghost traces intentionally keep is_active=True in the tree after
        # being evicted by _stop_lowest_running() — that decoupling must not block
        # the reset once the scheduler has fully drained between questions.
        real_active = len(self.scheduler.waiting) + len(self.scheduler.running)
        if real_active > 0:
            logger.warning(
                "[Gambit] reset_gambit_state called while %d scheduler requests are still active; skipping reset",
                real_active,
            )
            return

        # Clear sequence scorer GPU tensor history for all requests from the
        # completed question.  These tensors live on the GPU worker and are
        # never needed again once the question is done; failing to free them
        # leaks ~100-400 MB of VRAM per question and causes OOM after ~20 Qs.
        if self.trace_tree is not None:
            old_req_ids = list(self.trace_tree.nodes.keys())
            if old_req_ids:
                try:
                    self.collective_rpc(
                        "sync_sequence_histories",
                        args=(old_req_ids, []),
                    )
                except Exception:
                    pass  # non-sequence scorer builds: sync_sequence_histories is a no-op

        capacity = (
            self.gambit_config.tournament_capacity
            if self.gambit_config.tournament_mode
            else self.gambit_config.max_total_traces
        )
        self.trace_tree = TraceTree(
            max_traces=capacity,
            **self._trace_tree_branch_gates(),
        )
        self._gambit_output_token_ids.clear()
        self._seq_hs_to_clear.clear()
        self._seq_hs_to_copy.clear()

        # Also clear Gambit/score bookkeeping to keep memory bounded between calls.
        self.trace_scores.clear()
        self.trace_score_sums.clear()
        self.trace_score_counts.clear()
        self.final_trace_scores.clear()
        self.pending_low_score_stops.clear()
        self.pending_force_stops.clear()
        self.trace_token_counts.clear()
        self._stride_last_trigger.clear()
        self.pending_hs_classification.clear()
        self.trace_hidden_state_last.clear()

        # Restore the tournament cadence to its initial state. Without this the
        # `_tournament_last_round_step < 0` sentinel stays satisfied from the
        # previous prompt, so the warmup gate is skipped and the first round of
        # every subsequent prompt fires immediately on unscored traces.
        self._tournament_step_counter = 0
        self._tournament_last_round_step = -1


    def post_step(self, model_executed: bool) -> None:
        if self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            with record_function_or_nullcontext("core step_with_batch_queue: schedule"):
                scheduler_output = self.scheduler.schedule()
            with record_function_or_nullcontext(
                "core step_with_batch_queue: execute_model"
            ):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if scheduler_output.pending_structured_output_tokens:
                with record_function_or_nullcontext(
                    "core step_with_batch_queue: pending_structured_output_tokens"
                ):
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output
                    # Block-wait for execute to return
                    # (continues running async on the GPU).
                    with self.log_error_detail(scheduler_output):
                        exec_result = exec_future.result()
                        assert exec_result is None
            else:
                with record_function_or_nullcontext(
                    "core step_with_batch_queue: get_grammar_bitmask"
                ):
                    # We aren't waiting for any tokens, get any grammar
                    # output immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                # Block-wait for execute to return (continues running async on the GPU).
                with self.log_error_detail(scheduler_output):
                    exec_result = exec_future.result()

                if exec_result is None:
                    with record_function_or_nullcontext(
                        "core step_with_batch_queue: sample_tokens"
                    ):
                        # Call sample tokens.
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                else:
                    # No sampling required (e.g. all requests finished).
                    future = cast(Future[ModelRunnerOutput], exec_future)
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False
        with record_function_or_nullcontext("core step_with_batch_queue: model_output"):
            # Block until the next result is available.
            future, scheduler_output = batch_queue.pop()
            with self.log_error_detail(scheduler_output):
                model_output = future.result()

        # Prepare score histories for finished requests
        trace_score_histories_for_output = None
        if (
            self.gambit_config is not None
            and self.gambit_config.enable
            and self.gambit_config.record_score_history
        ):
            trace_score_histories_for_output = {
                rid: list(scores) for rid, scores in self.trace_scores.items()
            }

        with record_function_or_nullcontext(
            "core step_with_batch_queue: update_from_output"
        ):
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output,
                model_output,
                final_trace_scores=self.final_trace_scores,
                trace_score_histories=trace_score_histories_for_output,
            )
        if self.scheduler.finished_req_ids:
            self._finalize_scores(self.scheduler.finished_req_ids)

        if self.final_trace_scores:
            for eco in engine_core_outputs.values():
                for output in eco.outputs:
                    if output.finish_reason is not None and output.final_score is None:
                        output.final_score = self.final_trace_scores.get(
                            output.request_id
                        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            with record_function_or_nullcontext(
                "core step_with_batch_queue: deferred_scheduler_output"
            ):
                # We now have the tokens needed to compute the bitmask for the
                # deferred request. Get the bitmask and call sample tokens.
                grammar_output = self.scheduler.get_grammar_bitmask(
                    deferred_scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
                batch_queue.appendleft((future, deferred_scheduler_output))

        return engine_core_outputs, model_executed

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(
            method, timeout, args, kwargs, non_block=non_block
        )

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config, executor_class, log_stats, executor_fail_callback
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()