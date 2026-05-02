"""
Distributed generation helper for tensor-parallel inference.

Provides a unified :class:`Generator` interface that works transparently for
both single-GPU and tensor-parallel (multi-GPU via torchrun) modes.

In single-GPU mode (world_size=1), ``generate()`` runs the model directly.

In TP mode (world_size>1), rank 0 pushes work to an in-process queue, and a
collective worker loop (running on all ranks) pulls the work, broadcasts
input_ids to every rank via ``torch.distributed``, runs ``model.generate()``
in lockstep, and returns the result to rank 0.

Usage pattern (from the server entry point):

    gen = Generator(model, tokenizer, device)

    if gen.rank == 0:
        # Start HTTP server in a daemon thread; handlers call gen.generate()
        start_server_in_background(gen)

    # All ranks enter the worker loop. Rank 0 returns when shutdown() is
    # called; other ranks block here for the life of the process.
    gen.run_worker_loop()
"""

import json
import os
import queue
from typing import Any, Callable, Optional

import torch

# Worker-loop op codes (broadcast as an integer tensor to all ranks)
_OP_STOP = 0
_OP_GENERATE = 1
_OP_UPDATE_CONFIG = 2


def in_torchrun() -> bool:
    """True iff this process was launched with torchrun (has distributed env)."""
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


class Generator:
    """
    Unified generation interface for single-GPU and tensor-parallel modes.

    Attributes:
        rank:         This process's global rank (0 in single-GPU mode).
        world_size:   Total number of ranks (1 in single-GPU mode).
        distributed:  True if running under torchrun with world_size > 1.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Optional hook for applying runtime config updates (e.g. rebind
        # phase_configs on PhaseLayerAutoSwitch). Set via set_config_apply_fn().
        self._config_apply_fn: Optional[Callable[[dict], None]] = None

        if in_torchrun() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.distributed = True
            self._work_q: Optional[queue.Queue] = queue.Queue() if self.rank == 0 else None
            self._result_q: Optional[queue.Queue] = queue.Queue() if self.rank == 0 else None
        else:
            self.rank = 0
            self.world_size = 1
            self.distributed = False
            self._work_q = None
            self._result_q = None

    def set_config_apply_fn(self, fn: Callable[[dict], None]) -> None:
        """Register a callback that applies a config-update dict to local
        state on every rank. Called from the worker loop after receiving
        a broadcast of updated config. Typically binds to the switch
        object's ``phase_configs`` merge."""
        self._config_apply_fn = fn

    # ── Public API (called from HTTP handlers on rank 0) ────────────────────
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Run ``model.generate()``. In TP mode, dispatches via the worker loop.

        Must only be called from rank 0 (e.g. from a FastAPI handler).
        """
        assert self.rank == 0, "generate() must only be called from rank 0"

        if not self.distributed:
            return self._generate_direct(
                input_ids, attention_mask, max_new_tokens, temperature
            )

        # TP: push work and wait for the collective loop to pick it up
        self._work_q.put({
            "op":             _OP_GENERATE,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature":    temperature,
        })
        return self._result_q.get()

    def update_config(self, config_delta: dict) -> None:
        """Broadcast a partial config update to all ranks and apply it.

        In single-GPU mode, this just calls the registered apply fn on
        rank 0. In TP mode, the update is serialised, broadcast to all
        ranks via ``dist.broadcast``, and each rank applies it locally.

        ``config_delta`` is any JSON-serialisable dict. Its shape is
        determined by whatever ``set_config_apply_fn`` expects.
        """
        assert self.rank == 0, "update_config() must only be called from rank 0"

        if not self.distributed:
            if self._config_apply_fn is not None:
                self._config_apply_fn(config_delta)
            return

        # TP: queue the update and wait for the worker loop to confirm
        self._work_q.put({"op": _OP_UPDATE_CONFIG, "delta": config_delta})
        self._result_q.get()  # barrier: rank 0's apply finished

    def _generate_direct(self, input_ids, attention_mask, max_new_tokens, temperature):
        """Plain model.generate() — used directly in single-GPU mode or
        from inside the worker loop for a lockstepped call."""
        with torch.no_grad():
            return self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    # ── Collective worker loop (all ranks run this in TP mode) ──────────────
    def run_worker_loop(self) -> None:
        """
        Infinite loop that coordinates work across all ranks via op-codes.

        In single-GPU mode this is a no-op.

        In TP mode, all ranks enter this loop. Rank 0 blocks on its work
        queue (fed by HTTP handlers); other ranks block on
        ``dist.broadcast`` waiting for the op-code. Supported ops:

          _OP_STOP          — exit the loop (cleanly shut down)
          _OP_GENERATE      — broadcast input_ids, run generate() in lockstep
          _OP_UPDATE_CONFIG — broadcast a JSON config delta, apply locally

        Each iteration starts by broadcasting a single int op-code, then
        dispatches to the appropriate handler.
        """
        if not self.distributed:
            return  # nothing to coordinate

        import torch.distributed as dist

        while True:
            # ── Phase 1: pull next op from the queue (rank 0 only) ─────
            if self.rank == 0:
                work = self._work_q.get()  # blocks
                if work is None:
                    op = _OP_STOP
                else:
                    op = int(work["op"])
            else:
                work = None
                op = 0

            # ── Phase 2: broadcast the op code ─────────────────────────
            op_tensor = torch.tensor([op], dtype=torch.long, device=self.device)
            dist.broadcast(op_tensor, src=0)
            op = int(op_tensor.item())

            if op == _OP_STOP:
                return
            elif op == _OP_GENERATE:
                self._handle_generate(work)
            elif op == _OP_UPDATE_CONFIG:
                self._handle_update_config(work)
            else:
                if self.rank == 0:
                    print(f"[worker_loop] Unknown op {op}, ignoring")

    def _handle_generate(self, work: Optional[dict]) -> None:
        """Collective generate: broadcast inputs, run model.generate() on
        every rank, rank 0 posts result to its result queue."""
        import torch.distributed as dist

        # Broadcast metadata: [batch, seq_len, max_new_tokens]
        if self.rank == 0:
            ids = work["input_ids"]
            meta = torch.tensor(
                [ids.shape[0], ids.shape[1], int(work["max_new_tokens"])],
                dtype=torch.long, device=self.device,
            )
            temp_t = torch.tensor(
                [float(work["temperature"])],
                dtype=torch.float32, device=self.device,
            )
        else:
            meta = torch.zeros(3, dtype=torch.long, device=self.device)
            temp_t = torch.zeros(1, dtype=torch.float32, device=self.device)
        dist.broadcast(meta, src=0)
        dist.broadcast(temp_t, src=0)
        batch, seq_len, max_new_tokens = meta.tolist()
        temperature = float(temp_t.item())

        # Allocate & broadcast the actual tensors
        if self.rank == 0:
            input_ids = work["input_ids"].to(self.device)
            attention_mask = work["attention_mask"].to(self.device)
        else:
            input_ids = torch.zeros(
                batch, seq_len, dtype=torch.long, device=self.device
            )
            attention_mask = torch.ones(
                batch, seq_len, dtype=torch.long, device=self.device
            )
        dist.broadcast(input_ids, src=0)
        dist.broadcast(attention_mask, src=0)

        # All ranks run generate() in lockstep
        output = self._generate_direct(
            input_ids, attention_mask, max_new_tokens, temperature
        )

        if self.rank == 0:
            self._result_q.put(output)

    def _handle_update_config(self, work: Optional[dict]) -> None:
        """Collective config update: rank 0 serialises the delta dict to
        JSON bytes, broadcasts it, all ranks decode and apply locally."""
        import torch.distributed as dist

        # Serialise on rank 0
        if self.rank == 0:
            payload = json.dumps(work["delta"]).encode("utf-8")
            length = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
        else:
            length = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.broadcast(length, src=0)
        n = int(length.item())

        if self.rank == 0:
            buf = torch.tensor(list(payload), dtype=torch.uint8, device=self.device)
        else:
            buf = torch.zeros(n, dtype=torch.uint8, device=self.device)
        dist.broadcast(buf, src=0)

        delta = json.loads(bytes(buf.cpu().tolist()).decode("utf-8"))

        # Apply on every rank (local state mutation, no cross-rank op)
        if self._config_apply_fn is not None:
            self._config_apply_fn(delta)

        if self.rank == 0:
            self._result_q.put(None)  # ack

    def shutdown(self) -> None:
        """Signal the worker loop to exit cleanly (rank 0 only)."""
        if self.distributed and self.rank == 0:
            self._work_q.put(None)
