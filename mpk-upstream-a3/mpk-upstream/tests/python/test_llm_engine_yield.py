"""CPU-only tests for the LLMEngine yield/resume control flow."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "python" / "mirage" / "engine" / "llm_engine.py"


def _load_llm_engine():
    """Load LLMEngine with lightweight package stubs, without CUDA imports."""
    names = (
        "mirage",
        "mirage.engine",
        "mirage.engine.model_runner",
        "mirage.engine.tokenizer_manager",
        "mirage.mpk",
        "mirage.mpk.online_pinned_runtime",
        "mirage.engine.llm_engine",
    )
    previous = {name: sys.modules.get(name) for name in names}

    mirage = types.ModuleType("mirage")
    mirage.__path__ = []
    engine = types.ModuleType("mirage.engine")
    engine.__path__ = []
    mpk = types.ModuleType("mirage.mpk")
    mpk.__path__ = []
    model_runner = types.ModuleType("mirage.engine.model_runner")
    tokenizer_manager = types.ModuleType("mirage.engine.tokenizer_manager")
    runtime = types.ModuleType("mirage.mpk.online_pinned_runtime")

    class ModelRunner:
        pass

    class TokenizerManager:
        pass

    class OnlinePinnedRuntime:
        pass

    model_runner.ModelRunner = ModelRunner
    tokenizer_manager.TokenizerManager = TokenizerManager
    runtime.OnlinePinnedRuntime = OnlinePinnedRuntime
    sys.modules.update({
        "mirage": mirage,
        "mirage.engine": engine,
        "mirage.engine.model_runner": model_runner,
        "mirage.engine.tokenizer_manager": tokenizer_manager,
        "mirage.mpk": mpk,
        "mirage.mpk.online_pinned_runtime": runtime,
    })

    spec = importlib.util.spec_from_file_location(
        "mirage.engine.llm_engine", ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.LLMEngine


LLMEngine = _load_llm_engine()


class FakeRunner:
    def __init__(self, checkpoint_available: bool = True) -> None:
        self.kernel_active = threading.Event()
        self.kernel_active.set()
        self.checkpoint_available = checkpoint_available
        self.yield_requests = 0
        self.resumes = 0

    def request_yield(self) -> None:
        self.yield_requests += 1
        self.kernel_active.clear()

    def resume(self) -> None:
        self.resumes += 1
        self.kernel_active.set()


def _engine(runner: FakeRunner):
    engine = LLMEngine.__new__(LLMEngine)
    engine.model_runner = runner
    engine._submit_lock = threading.RLock()
    engine._kernel_control_lock = threading.Lock()
    engine._kernel_launched = threading.Event()
    engine._kernel_launched.set()
    engine._kernel_thread = None
    engine._kernel_error = None
    engine._yielded = False
    engine._yielding = False
    engine._closed = False
    engine.tokenizer_manager = types.SimpleNamespace(
        tokenize=lambda prompt, use_template: [1])
    engine._next_rid = 0
    return engine


class LLMEngineYieldTests(unittest.TestCase):
    def test_yield_blocks_submits_and_resume_restarts_kernel(self) -> None:
        runner = FakeRunner()
        engine = _engine(runner)

        engine.yield_kernel()
        self.assertTrue(engine.is_yielded)
        self.assertEqual(runner.yield_requests, 1)
        with self.assertRaisesRegex(RuntimeError, "yielded"):
            engine.submit("prompt")

        engine.resume()
        engine._kernel_thread.join(timeout=2)
        self.assertFalse(engine.is_yielded)
        self.assertEqual(runner.resumes, 1)
        self.assertTrue(runner.kernel_active.is_set())

    def test_yield_rejects_missing_checkpoint(self) -> None:
        engine = _engine(FakeRunner(checkpoint_available=False))
        with self.assertRaisesRegex(RuntimeError, "without a resumable checkpoint"):
            engine.yield_kernel()
        self.assertFalse(engine.is_yielded)


if __name__ == "__main__":
    unittest.main()
