"""Auto-register adapters by import."""
from . import base  # noqa: F401
from . import daily_omni  # noqa: F401
from . import omnibench  # noqa: F401
from . import omnivideobench  # noqa: F401
from . import unified  # noqa: F401  # 真正注册 omnibench/daily_omni/omnivideobench 的地方
