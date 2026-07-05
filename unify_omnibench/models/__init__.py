"""Auto-register models on import. Heavy backends are imported lazily
inside each model's `load()`, so simply importing this package is cheap.
"""
from . import base  # noqa: F401
from . import echo  # noqa: F401

# API models (cheap to import; SDK loaded at .load())
try:
    from .api import openai_chat  # noqa: F401
except Exception:  # pragma: no cover
    pass
try:
    from .api import gemini  # noqa: F401
except Exception:  # pragma: no cover
    pass

# Local models — import path only; heavy deps in load()
try:
    from .local import qwen25omni  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:
    from .vllm_backend import vllm_runner  # noqa: F401
except Exception:  # pragma: no cover
    pass
