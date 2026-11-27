"""LLMEngine — concurrent serving loop backed on persistent kernel + ring buffer.
"""

from __future__ import annotations

import queue
import threading
import time

import torch

from .model_runner import ModelRunner
from .tokenizer_manager import TokenizerManager
from ..mpk.online_pinned_runtime import OnlinePinnedRuntime


class _StreamingMonitor:
    """Single background thread that monitors all active streaming sessions.

    Replaces the per-request polling thread (Thread B) with one shared daemon
    that polls completion state and step progress, then enqueues tokens for
    every registered session. This eliminates the thread explosion that
    causes GIL contention under concurrent load.
    """

    def __init__(self, runtime: OnlinePinnedRuntime, tokenizer_manager: TokenizerManager) -> None:
        self._runtime = runtime
        self._tokenizer_manager = tokenizer_manager
        self._sessions: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register(self, rid: int, prompt_len: int, timeout: float) -> queue.Queue:
        """Register a streaming session and return its token queue."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._sessions[rid] = {
                'q': q,
                'prompt_len': prompt_len,
                'row': -1,
                'last_step': prompt_len - 1,
                'deadline': time.monotonic() + timeout,
            }
        return q

    def unregister(self, rid: int) -> None:
        """Remove a session whose request could not be published."""
        with self._lock:
            self._sessions.pop(rid, None)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    for rid, s in list(self._sessions.items()):
                        released = False
                        try:
                            # Completion is authoritative and may arrive before
                            # the monitor observes the transient row mapping.
                            completion = self._runtime.get_completion(rid)
                            if completion is not None:
                                row, final_step = completion
                                try:
                                    self._yield_remaining(s, row, final_step)
                                finally:
                                    released = self._runtime.release_request(rid)
                                if not released:
                                    raise RuntimeError(
                                        f"missing completion for rid={rid}")
                                del self._sessions[rid]
                                continue

                            # Phase 1 — discover buffer row.
                            if s['row'] == -1:
                                row = self._runtime.find_row_for_rid(rid)
                                if row >= 0:
                                    s['row'] = row
                                elif time.monotonic() > s['deadline']:
                                    self._runtime.abandon_request(rid)
                                    s['q'].put(("__timeout__", True))
                                    del self._sessions[rid]
                                continue

                            row = s['row']

                            # Phase 2 — poll step progress, read only new tokens.
                            current_step = self._runtime.get_current_step_at_row(row)
                            if current_step > s['last_step']:
                                new_tokens = self._runtime.read_tokens_range(
                                    row, s['last_step'] + 1, current_step)
                                for tid in new_tokens.tolist():
                                    text = self._tokenizer_manager.decode_single(tid)
                                    s['last_step'] += 1
                                    s['q'].put((text, False))

                            # Check timeout.
                            if time.monotonic() > s['deadline']:
                                self._runtime.abandon_request(rid)
                                s['q'].put(("__timeout__", True))
                                del self._sessions[rid]

                        except Exception:
                            try:
                                if not released:
                                    self._runtime.abandon_request(rid)
                            finally:
                                s['q'].put(("__error__", True))
                                del self._sessions[rid]

            except Exception:
                pass

            self._stop.wait(0.002)  # 2 ms polling interval

    def _yield_remaining(self, s: dict, row: int, final_step: int) -> None:
        """Yield all remaining tokens for a completed request."""
        if final_step > s['last_step']:
            new_tokens = self._runtime.read_tokens_range(
                row, s['last_step'] + 1, final_step)
            new_ids = new_tokens.tolist()
            for j, tid in enumerate(new_ids):
                text = self._tokenizer_manager.decode_single(tid)
                is_final = (j == len(new_ids) - 1)
                s['q'].put((text, is_final))
                s['last_step'] += 1
        else:
            s['q'].put(("", True))

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join()
        with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
        cleanup_error: Exception | None = None
        for rid, session in sessions:
            # The GPU has no per-request cancellation. Mark each request so
            # the runtime releases its row if a completion arrives later, and
            # always wake the user-facing generator.
            try:
                self._runtime.abandon_request(rid)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                session['q'].put(("__closed__", True))
        if cleanup_error is not None:
            raise RuntimeError(
                "failed to abandon a streaming request") from cleanup_error


class LLMEngine:
    """Generation loop backed by the ``online_pinned`` persistent kernel.

    The kernel runs in a background thread so the engine can accept requests
    concurrently.  Each call to :meth:`submit` allocates a unique, never-
    repeating *request id* (rid), stages tokens in the pinned inbox, writes a
    ring-buffer entry, and then blocks until the GPU reports completion for
    that specific rid.  The GPU manages its own buffer-row pool and a
    waiting/running queue pair, so the CPU does not need to reason about
    slot availability.

    Args:
        model_runner: A fully constructed :class:`ModelRunner` whose MPK is
                      compiled in ``online_pinned`` mode.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.runtime: OnlinePinnedRuntime = model_runner.runtime
        self.tokenizer_manager = TokenizerManager(model_runner.tokenizer)

        # Monotonically incrementing request id (never wraps).
        self._next_rid: int = 0

        # Serialises ring-buffer writes so two callers do not interleave.
        self._submit_lock = threading.RLock()
        # Serialises yield/resume/close transitions, which may block on GPU work.
        self._kernel_control_lock = threading.Lock()

        # Shared streaming monitor — replaces per-request polling threads.
        self._monitor: _StreamingMonitor

        # Background kernel bookkeeping.
        self._kernel_launched: threading.Event = threading.Event()
        self._kernel_thread: threading.Thread | None = None
        self._kernel_error: BaseException | None = None
        self._yielded = False
        self._yielding = False
        self._closed = False

        # Launch MPK kernels
        self._ensure_kernel_running()
        self._monitor = _StreamingMonitor(self.runtime, self.tokenizer_manager)

    # ── Public API ────────────────────────────────────────────────────────

    def submit(
        self,
        prompt: str,
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
        stream: bool = False,
    ):
        """Submit a single prompt for generation.

        Safe to call concurrently — each invocation gets a unique rid and
        serialises the ring-buffer write under an internal lock.

        Args:
            prompt:        String prompt.
            use_template:  Apply chat template before tokenizing.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.
            stream:        If True, returns a generator yielding ``(text,
                           is_final)`` tuples. Otherwise returns a dict.

        Returns:
            When stream=False: ``{"text": str, "token_ids": list[int]}``
            When stream=True:  generator yielding ``(text, is_final)``
        """
        token_ids = self.tokenizer_manager.tokenize(prompt, use_template)
        prompt_len = len(token_ids)

        t = torch.tensor(token_ids, dtype=torch.int64)
        stream_queue: queue.Queue | None = None
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("LLMEngine is closed")
            if self._yielded or self._yielding:
                raise RuntimeError("LLMEngine is yielded or currently draining")
            self._raise_kernel_error()
            rid = self._next_rid
            self._next_rid += 1
            if stream:
                stream_queue = self._monitor.register(
                    rid, prompt_len, timeout)
            try:
                self.runtime.submit(rid, t)
            except Exception:
                if stream:
                    self._monitor.unregister(rid)
                raise

        if stream:
            assert stream_queue is not None
            return self._submit_stream(rid, stream_queue)
        else:
            buffer_row, final_step = self.runtime.wait_for_request(
                rid, timeout, poll_interval)
            try:
                full_tokens = self.runtime.read_tokens_at_row(
                    buffer_row, final_step)
                output_ids = full_tokens[prompt_len:].tolist()
                return {
                    "text": self.tokenizer_manager.decode(output_ids),
                    "token_ids": output_ids,
                }
            finally:
                self.runtime.release_request(rid)

    def yield_kernel(self) -> None:
        """Drain the MPK worker/scheduler loops and preserve active decode state.

        The call blocks until the GPU is safe to hand off. In distributed mode,
        invoke it on every MPI rank. Existing requests remain paused and can
        continue after :meth:`resume`.
        """
        with self._kernel_control_lock:
            with self._submit_lock:
                if self._closed:
                    raise RuntimeError("LLMEngine is closed")
                if self._yielded or self._yielding:
                    raise RuntimeError("LLMEngine is already yielded or draining")
                self._yielding = True
            try:
                self._wait_for_kernel_launch()
                self.model_runner.request_yield()
                if self._kernel_thread is not None:
                    self._kernel_thread.join()
                self._raise_kernel_error()
                if not self.model_runner.checkpoint_available:
                    raise RuntimeError(
                        "MPK stopped without a resumable checkpoint")
                with self._submit_lock:
                    self._yielded = True
                    self._yielding = False
            except BaseException:
                with self._submit_lock:
                    self._yielding = False
                raise

    def resume(self) -> None:
        """Resume decoding from the in-memory MPK checkpoint."""
        with self._kernel_control_lock:
            with self._submit_lock:
                if self._closed:
                    raise RuntimeError("LLMEngine is closed")
                if not self._yielded:
                    raise RuntimeError("LLMEngine has no yielded checkpoint")
                self._yielding = True
            try:
                self._start_kernel_thread(resume=True)
                self._wait_for_kernel_launch()
                with self._submit_lock:
                    self._yielded = False
                    self._yielding = False
            except BaseException:
                with self._submit_lock:
                    self._yielding = False
                raise

    @property
    def is_yielded(self) -> bool:
        with self._submit_lock:
            return self._yielded or self._yielding

    # ── Internal ──────────────────────────────────────────────────────────

    def _wait_for_kernel_launch(self) -> None:
        while not self.model_runner.kernel_active:
            thread = self._kernel_thread
            if thread is None or not thread.is_alive():
                self._raise_kernel_error()
                raise RuntimeError("MPK kernel is not running")
            time.sleep(0.001)

    def _start_kernel_thread(self, resume: bool = False) -> None:
        self._kernel_error = None
        self._kernel_thread = threading.Thread(
            target=self._run_kernel,
            args=(resume,),
            daemon=True,
        )
        self._kernel_thread.start()

    def _run_kernel(self, resume: bool) -> None:
        try:
            if resume:
                self.model_runner.resume()
            else:
                self.model_runner()
        except BaseException as exc:
            self._kernel_error = exc

    def _raise_kernel_error(self) -> None:
        if self._kernel_error is not None:
            raise RuntimeError("MPK kernel thread failed") from self._kernel_error

    def _ensure_kernel_running(self) -> None:
        """Launch the persistent kernel once in a background daemon thread."""
        if self._kernel_launched.is_set():
            return
        with self._submit_lock:
            if self._kernel_launched.is_set():
                return
            self.runtime.reset()
            self.runtime.start()
            self._start_kernel_thread(resume=False)
            self._kernel_launched.set()

    def _submit_stream(
        self,
        rid: int,
        q: queue.Queue,
    ):
        """Generator: yield ``(text, is_final)`` as tokens are decoded.

        Registers with the shared :class:`_StreamingMonitor` instead of
        spawning a dedicated polling thread, so the number of polling
        threads stays constant regardless of concurrent request count.
        """
        def generator():
            while True:
                try:
                    text, is_final = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if text == "__timeout__":
                    raise TimeoutError(
                        f"stream timed out for rid={rid}")
                if text == "__error__":
                    raise RuntimeError(
                        f"stream error for rid={rid}")
                if text == "__closed__":
                    raise RuntimeError(
                        f"engine closed while streaming rid={rid}")
                yield (text, is_final)
                if is_final:
                    break

        return generator()

    def close(self) -> None:
        """Signal the GPU kernel to shut down at the next idle cycle."""
        with self._kernel_control_lock:
            with self._submit_lock:
                if self._closed:
                    return
                self._closed = True
                was_yielded = self._yielded
                self._yielding = was_yielded
            monitor_error: Exception | None = None
            try:
                self._monitor.shutdown()
            except Exception as exc:
                monitor_error = exc
            try:
                self.runtime.request_shutdown()
                if was_yielded:
                    # The shutdown flag is visible when the saved scheduler
                    # reaches its next batch boundary after resuming.
                    self._start_kernel_thread(resume=True)
                    with self._submit_lock:
                        self._yielded = False
                        self._yielding = False
                if self._kernel_thread is not None:
                    self._kernel_thread.join()
                self._raise_kernel_error()
            finally:
                self.runtime.stop()
            if monitor_error is not None:
                raise monitor_error
