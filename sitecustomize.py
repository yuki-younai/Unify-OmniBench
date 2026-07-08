"""Auto-imported by EVERY Python process that has this directory on
``PYTHONPATH`` (Python's ``site`` module imports ``sitecustomize`` at
interpreter startup — see ``eval.sh``'s ``export PYTHONPATH=...``).

This exists so the compat shim below also reaches vLLM's worker
subprocesses (spawned via ``VLLM_WORKER_MULTIPROC_METHOD=spawn``), which
start a brand-new interpreter and re-import ``transformers`` from scratch —
an in-memory monkeypatch applied only in the main process (e.g. inside
``vllm_runner.py::load()``) would not reach them.

Shim: some transformers builds removed the legacy
``all_special_tokens_extended`` property that older vLLM tokenizer-caching
code still reads at startup (``AttributeError: Qwen2Tokenizer has no
attribute all_special_tokens_extended``). It only ever differed from
``all_special_tokens`` by possibly containing ``AddedToken`` objects
instead of plain strings, irrelevant here, so falling back to
``all_special_tokens`` is a safe equivalent. See
``docs/Unify-OmniBench-v0.1.0-dev.md`` for the full debugging history.
"""
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    # Never let this shim itself break startup — best-effort only.
    pass
