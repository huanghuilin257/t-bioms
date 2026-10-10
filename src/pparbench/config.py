"""Documentation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ppar_1/src/pparbench/config.py → parents[2] = ppar_1
PPAR1_ROOT = Path(__file__).resolve().parents[2]

PPAR_ROOT = PPAR1_ROOT.parent

CONFIG_DIR = PPAR1_ROOT / "configs"
DATA_DIR = PPAR1_ROOT / "data"
RESULTS_DIR = PPAR1_ROOT / "results"
FIGURES_DIR = PPAR1_ROOT / "figures"


class ConfigError(ValueError):
    """Documentation."""


@dataclass
class TargetConfig:
    slug: str
    name: str
    chembl_target_id: str
    endpoint: str
    label_column: str
    source: str                      # curated_csv | chembl_export
    paths: dict[str, Path]
    cutpoints: list[int] | None
    curation: dict[str, Any] = field(default_factory=dict)
    uniprot: str | None = None
    protein_sequence: str | None = None
    notes: str = ""

    @property
    def curated_csv(self) -> Path | None:
        return self.paths.get("curated_csv")

    @property
    def chembl_export(self) -> Path | None:
        return self.paths.get("chembl_export")


@dataclass
class ProtocolConfig:
    seed: int
    n_jobs: int
    features: dict[str, Any]
    protocols: dict[str, Any]
    models: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    bootstrap: dict[str, Any]
    applicability_domain: dict[str, Any]
    omp_num_threads: int = 1
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def fingerprint(self) -> str:
        """Documentation."""
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _require(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if d.get(k) in (None, "")]
    if missing:
        raise ConfigError(f"{where} : {', '.join(missing)}")


def load_protocol(path: Path | None = None) -> ProtocolConfig:
    path = path or CONFIG_DIR / "protocol.yaml"
    if not path.exists():
        raise ConfigError(f"missingconfiguration: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(raw, ["seed", "features", "protocols", "models", "metrics",
                   "bootstrap", "applicability_domain"], f"protocol.yaml ({path})")
    return ProtocolConfig(
        seed=int(raw["seed"]),
        n_jobs=int(raw.get("n_jobs", 8)),
        features=raw["features"],
        protocols=raw["protocols"],
        models=raw["models"],
        metrics=raw["metrics"],
        bootstrap=raw["bootstrap"],
        applicability_domain=raw["applicability_domain"],
        omp_num_threads=int(raw.get("omp_num_threads", 1)),
        raw=raw,
    )


def load_target(slug: str) -> TargetConfig:
    if slug.startswith("_"):
        raise ConfigError(f"'{slug}' target _template.yaml")
    path = CONFIG_DIR / "targets" / f"{slug}.yaml"
    if not path.exists():
        available = sorted(
            p.stem for p in (CONFIG_DIR / "targets").glob("*.yaml")
            if not p.stem.startswith("_")
        )
        raise ConfigError(
            f"missingtargetconfiguration {path}\navailabletarget: {', '.join(available) or '()'}\n"
            f"target configs/targets/_template.yaml"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _require(raw, ["slug", "chembl_target_id", "endpoint", "source"], f"{path}")

    if raw["slug"] != slug:
        raise ConfigError(f"{path} slug='{raw['slug']}' '{slug}' inconsistent")
    if raw["source"] not in ("curated_csv", "chembl_export"):
        raise ConfigError(f"{path}: source curated_csv chembl_export"
                          f"'{raw['source']}'")

    
    
    
    
    
    
    paths: dict[str, Path] = {}
    for key, val in (raw.get("paths") or {}).items():
        if val:
            paths[key] = (PPAR_ROOT / val).resolve()

    if raw["source"] == "curated_csv" and "curated_csv" not in paths:
        raise ConfigError(f"{path}: source=curated_csv paths.curated_csv")
    if raw["source"] == "chembl_export" and "chembl_export" not in paths:
        raise ConfigError(f"{path}: source=chembl_export paths.chembl_export")
    
    if "chembl_export" not in paths:
        raise ConfigError(f"{path}: paths.chembl_export temporalyearsource")

    return TargetConfig(
        slug=raw["slug"],
        name=raw.get("name") or raw["slug"],
        chembl_target_id=raw["chembl_target_id"],
        endpoint=raw["endpoint"],
        label_column=raw.get("label_column", "pchembl_mean"),
        source=raw["source"],
        paths=paths,
        cutpoints=raw.get("cutpoints"),
        curation=raw.get("curation") or {},
        uniprot=raw.get("uniprot"),
        protein_sequence=raw.get("protein_sequence"),
        notes=raw.get("notes", ""),
    )


def list_targets() -> list[str]:
    return sorted(
        p.stem for p in (CONFIG_DIR / "targets").glob("*.yaml")
        if not p.stem.startswith("_")
    )
