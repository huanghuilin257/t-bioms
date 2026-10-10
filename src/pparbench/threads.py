"""Documentation."""
from __future__ import annotations

import os

_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_thread_limits(n: int | None = None) -> int:
    """Documentation."""
    if n is None:
        externally_set = {v: os.environ.get(v) for v in _VARS}
        if any(externally_set.values()):
            return int(next(v for v in externally_set.values() if v))
        
        import yaml

        from .config import CONFIG_DIR
        raw = yaml.safe_load((CONFIG_DIR / "protocol.yaml").read_text(encoding="utf-8"))
        n = int(raw.get("omp_num_threads", 1))
    for v in _VARS:
        os.environ[v] = str(n)
    return int(n)
