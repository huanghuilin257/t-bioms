"""Documentation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import CANONICAL_COLUMNS
from .config import ConfigError, TargetConfig
from .features import canonical_smiles


COL_MOL = "Molecule ChEMBL ID"
COL_SMILES = "Smiles"
COL_TYPE = "Standard Type"
COL_REL = "Standard Relation"
COL_PCHEMBL = "pChEMBL Value"
COL_ASSAY_TYPE = "Assay Type"
COL_ASSAY_DESC = "Assay Description"
COL_TARGET = "Target ChEMBL ID"
COL_YEAR = "Document Year"
COL_ORGANISM = "Assay Organism"


COL_TARGET_ORGANISM = "Target Organism"
COL_VALIDITY = "Data Validity Comment"


def _organism_column(df: pd.DataFrame) -> str | None:
    """Documentation."""
    for c in (COL_ORGANISM, COL_TARGET_ORGANISM):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def read_chembl_export(path) -> pd.DataFrame:
    """Documentation."""
    head = open(path, "r", encoding="utf-8", errors="replace").readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    missing = [c for c in (COL_MOL, COL_SMILES, COL_YEAR) if c not in df.columns]
    if missing:
        raise ConfigError(f"{path} ChEMBL missing columns: {missing}")
    return df


def year_lookup(chembl_export_path) -> pd.Series:
    """Documentation."""
    df = read_chembl_export(chembl_export_path)
    yr = pd.to_numeric(df[COL_YEAR], errors="coerce")
    return (
        pd.DataFrame({"chembl_id": df[COL_MOL], "year": yr})
        .dropna(subset=["chembl_id"])
        .groupby("chembl_id")["year"]
        .min()
    )



def from_curated_csv(tcfg: TargetConfig) -> pd.DataFrame:
    """Documentation."""
    df = pd.read_csv(tcfg.curated_csv)
    if tcfg.label_column not in df.columns:
        raise ConfigError(
            f"{tcfg.curated_csv} '{tcfg.label_column}'"
            f": {list(df.columns)}"
        )
    if "chembl_id" not in df.columns:
        raise ConfigError(f"{tcfg.curated_csv} chembl_id join year")

    out = pd.DataFrame({
        "chembl_id": df["chembl_id"].astype(str),
        "smiles": df["smiles"].astype(str),
        "y": pd.to_numeric(df[tcfg.label_column], errors="coerce"),
    })
    
    out["n_obs"] = (
        pd.to_numeric(df["n_obs"], errors="coerce") if "n_obs" in df.columns
        else df["raw_values"].astype(str).str.count(r"[;,]") + 1
        if "raw_values" in df.columns else np.nan
    )
    for cand in ("pchembl_std", "y_std"):
        if cand in df.columns:
            out["y_std"] = pd.to_numeric(df[cand], errors="coerce")
            break
    else:
        out["y_std"] = np.nan

    out["year"] = out["chembl_id"].map(year_lookup(tcfg.chembl_export))
    return _finalize(out, tcfg)



def from_chembl_export(tcfg: TargetConfig) -> pd.DataFrame:
    """Documentation."""
    cur = tcfg.curation
    df = read_chembl_export(tcfg.chembl_export)
    log: list[dict] = []

    def step(name: str, keep_mask) -> None:
        nonlocal df
        before = len(df)
        df = df[keep_mask].copy()
        log.append({"step": name, "rows_before": before, "rows_after": len(df),
                    "dropped": before - len(df)})

    log.append({"step": "load", "rows_before": len(df), "rows_after": len(df), "dropped": 0})

    
    if COL_TARGET in df.columns:
        step(f"target=={tcfg.chembl_target_id}", df[COL_TARGET] == tcfg.chembl_target_id)

    
    step(f"endpoint=={tcfg.endpoint}", df[COL_TYPE].astype(str).str.upper() == tcfg.endpoint.upper())

    
    at = cur.get("assay_types")
    if at and COL_ASSAY_TYPE in df.columns:
        step(f"assay_type∈{at}", df[COL_ASSAY_TYPE].isin(at))

    if cur.get("organism"):
        oc = _organism_column(df)
        if oc is None:
            
            
            print(f"[curate] !! configuration organism=={cur['organism']}"
                  f"{COL_ORGANISM} / {COL_TARGET_ORGANISM} available missing columnsempty column"
                  "skip target Target ChEMBL ID")
        else:
            step(f"{oc}=={cur['organism']}", df[oc] == cur["organism"])

    if cur.get("drop_validity_flagged", True) and COL_VALIDITY in df.columns:
        step("validity", df[COL_VALIDITY].isna())

    
    if cur.get("require_exact_relation", True) and COL_REL in df.columns:
        step("relation=='='", df[COL_REL].astype(str).str.strip().isin(["=", "'='"]))

    if cur.get("exclude_assay_regex") and COL_ASSAY_DESC in df.columns:
        pat = re.compile(cur["exclude_assay_regex"], re.I)
        step("assay",
             ~df[COL_ASSAY_DESC].astype(str).str.contains(pat, na=False))

    step("pChEMBL", pd.to_numeric(df[COL_PCHEMBL], errors="coerce").notna())
    step("SMILES", df[COL_SMILES].notna() & (df[COL_SMILES].astype(str) != ""))

    
    df["_p"] = pd.to_numeric(df[COL_PCHEMBL], errors="coerce")
    df["_yr"] = pd.to_numeric(df[COL_YEAR], errors="coerce")
    g = df.groupby(COL_MOL).agg(
        smiles=(COL_SMILES, "first"),
        y=("_p", "mean"),
        y_std=("_p", "std"),
        n_obs=("_p", "size"),
        year=("_yr", "min"),
    ).reset_index().rename(columns={COL_MOL: "chembl_id"})
    log.append({"step": "molecules", "rows_before": len(df), "rows_after": len(g),
                "dropped": len(df) - len(g)})

    before = len(g)
    thr = cur.get("max_pchembl_std")
    if thr is not None:
        g = g[(g["n_obs"] == 1) | (g["y_std"].fillna(0) <= float(thr))].copy()
        log.append({"step": f"std<={thr}", "rows_before": before,
                    "rows_after": len(g), "dropped": before - len(g)})

    before = len(g)
    min_obs = int(cur.get("min_observations", 1))
    if min_obs > 1:
        g = g[g["n_obs"] >= min_obs].copy()
        log.append({"step": f"n_obs>={min_obs}", "rows_before": before,
                    "rows_after": len(g), "dropped": before - len(g)})

    out = _finalize(g, tcfg)
    out.attrs["curation_log"] = pd.DataFrame(log)
    return out



def _finalize(df: pd.DataFrame, tcfg: TargetConfig) -> pd.DataFrame:
    """Documentation."""
    df = df.dropna(subset=["smiles", "y"]).copy()
    df["smiles_canonical"] = canonical_smiles(df["smiles"].astype(str).tolist())
    n_bad = int(df["smiles_canonical"].isna().sum())
    df = df.dropna(subset=["smiles_canonical"]).copy()

    n_before = len(df)
    
    agg = df.groupby("smiles_canonical").agg(
        chembl_id=("chembl_id", "first"),
        y=("y", "mean"),
        y_std=("y_std", "mean"),
        n_obs=("n_obs", "sum"),
        year=("year", "min"),
    ).reset_index().rename(columns={"smiles_canonical": "smiles"})
    n_dup = n_before - len(agg)

    
    agg = agg.sort_values(["year", "chembl_id"], na_position="last").reset_index(drop=True)
    agg["mol_id"] = [f"{tcfg.slug}_{i:05d}" for i in range(len(agg))]

    out = agg[CANONICAL_COLUMNS].copy()
    out.attrs["finalize_log"] = {
        "invalid_smiles_dropped": n_bad,
        "duplicate_structures_merged": n_dup,
        "n_final": len(out),
        "n_missing_year": int(out["year"].isna().sum()),
        "year_min": None if out["year"].isna().all() else int(out["year"].min()),
        "year_max": None if out["year"].isna().all() else int(out["year"].max()),
    }
    return out


def build_dataset(tcfg: TargetConfig) -> pd.DataFrame:
    if tcfg.source == "curated_csv":
        return from_curated_csv(tcfg)
    return from_chembl_export(tcfg)
