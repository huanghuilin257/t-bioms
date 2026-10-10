"""Documentation."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, FIGURES_DIR, PPAR1_ROOT, RESULTS_DIR


def target_data_dir(slug: str, create: bool = True) -> Path:
    p = DATA_DIR / slug
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def target_results_dir(slug: str, stage: str, create: bool = True) -> Path:
    p = RESULTS_DIR / slug / stage
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(slug: str | None = None, create: bool = True) -> Path:
    p = FIGURES_DIR / slug if slug else FIGURES_DIR
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def canonical_dataset_path(slug: str) -> Path:
    """Documentation."""
    return target_data_dir(slug) / "dataset.csv"


def features_path(slug: str, kind: str = "ligand2057") -> Path:
    return target_data_dir(slug) / f"features_{kind}.npz"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PPAR1_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _pkg_versions() -> dict[str, str]:
    vers: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "rdkit"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            vers[mod] = "missing"
    return vers


def write_run_meta(
    outdir: Path,
    *,
    target: str,
    stage: str,
    protocol_fingerprint: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Documentation."""
    meta = {
        "target": target,
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "protocol_fingerprint": protocol_fingerprint,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "host": platform.node(),
        "versions": _pkg_versions(),
    }
    if extra:
        meta.update(extra)
    path = outdir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame, path: Path, *, float_fmt: str = "%.4f") -> Path:
    """Documentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_fmt)
    return path
