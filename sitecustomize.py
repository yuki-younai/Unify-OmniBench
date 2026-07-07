"""Auto-imported by EVERY Python process that has this directory on
``PYTHONPATH`` (Python's ``site`` module imports ``sitecustomize`` at
interpreter startup if found on ``sys.path`` — see ``eval.sh``'s
``export PYTHONPATH=...``).

This exists specifically so the fix below also reaches vLLM's
worker subprocesses (spawned via ``VLLM_WORKER_MULTIPROC_METHOD=spawn``),
which start a brand-new interpreter and re-import ``transformers`` from
scratch — an in-memory monkeypatch applied only in the main process
(e.g. inside ``vllm_runner.py::load()``) does NOT reach them.

[2026-07-07] transformers/vllm version-compat shim
---------------------------------------------------
The transformers build pinned for Qwen2.5-Omni support (a newer/dev
commit — see model loading code comments about
``pip install git+...@3a1ead0...``) removed the legacy
``all_special_tokens_extended`` property that pinned ``vllm==0.11.0``'s
own tokenizer-caching code (``vllm/transformers_utils/tokenizer.py::
get_cached_tokenizer``) still reads at engine/worker startup:

    AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

This has been observed failing in TWO places:
  1. Main process, inside ``LLMEngine.__init__`` -> ``init_tokenizer_from_configs``.
  2. vLLM worker subprocess, inside ``WorkerProc.__init__`` ->
     ``worker_receiver_cache_from_config`` -> ... -> ``get_cached_tokenizer``.

``all_special_tokens_extended`` historically differs from
``all_special_tokens`` only in that it may contain ``AddedToken`` objects
instead of plain strings -- irrelevant for vLLM's use here (just
snapshotting special tokens for its tokenizer cache), so falling back to
``all_special_tokens`` is a safe, semantically-equivalent shim.
"""
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    # Never let this shim itself break startup (e.g. if transformers isn't
    # installed yet, or its internals changed again) -- it's best-effort.
    pass
