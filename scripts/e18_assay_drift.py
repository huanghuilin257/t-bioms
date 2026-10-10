#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()          

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402
from scipy.stats import wilcoxon                        # noqa: E402

from pparbench.config import load_protocol, load_target             # noqa: E402
from pparbench.io import target_results_dir, save_table, write_run_meta  # noqa: E402
from pparbench import curate as CU                                  # noqa: E402

DECLARED_IN = "manuscript protocol declaration"

SCAFFOLD = "scaffold_cv"
TEMPORAL = "temporal"
POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]     


FROZEN20 = [
    "abl1", "ache", "adora2a", "ar", "bace1", "drd2", "egfr", "esr1", "f10", "f2",
    "hdac1", "htr2a", "kcnh2", "kdr", "nr3c1", "oprm1", "pik3ca", "ppara", "ppard", "pparg",
]



SUPERSET_SLUGS = {"pparg"}

COL_DOC = "Document ChEMBL ID"
COL_ASSAY_ID = "Assay ChEMBL ID"        
MIN_SEEN_N = 50                          

N_BOOT = 2000
BOOT_SEED = 20260825
CONSERVATIVE = 1.222                     



def _filtered_activities(tcfg) -> pd.DataFrame:
    """Documentation."""
    df = CU.read_chembl_export(tcfg.chembl_export)
    cur = tcfg.curation

    if CU.COL_TARGET in df.columns:
        df = df[df[CU.COL_TARGET] == tcfg.chembl_target_id]
    df = df[df[CU.COL_TYPE].astype(str).str.upper() == tcfg.endpoint.upper()]

    at = cur.get("assay_types")
    if at and CU.COL_ASSAY_TYPE in df.columns:
        df = df[df[CU.COL_ASSAY_TYPE].isin(at)]

    if cur.get("organism"):
        oc = CU._organism_column(df)
        if oc is not None:
            df = df[df[oc] == cur["organism"]]

    if cur.get("drop_validity_flagged", True) and CU.COL_VALIDITY in df.columns:
        df = df[df[CU.COL_VALIDITY].isna()]

    if cur.get("require_exact_relation", True) and CU.COL_REL in df.columns:
        df = df[df[CU.COL_REL].astype(str).str.strip().isin(["=", "'='"])]

    if cur.get("exclude_assay_regex") and CU.COL_ASSAY_DESC in df.columns:
        import re
        pat = re.compile(cur["exclude_assay_regex"], re.I)
        df = df[~df[CU.COL_ASSAY_DESC].astype(str).str.contains(pat, na=False)]

    df = df[pd.to_numeric(df[CU.COL_PCHEMBL], errors="coerce").notna()]
    df = df[df[CU.COL_SMILES].notna() & (df[CU.COL_SMILES].astype(str) != "")]
    return df


def assay_map(slug: str, dataset: pd.DataFrame, *, key: str = "proxy"):
    """Documentation."""
    tcfg = load_target(slug)
    exp = tcfg.chembl_export
    if exp is None or not Path(exp).exists():
        raise FileNotFoundError(f"{slug} : {exp}")

    if slug in SUPERSET_SLUGS:
        
        df = CU.read_chembl_export(exp)
        mode = "superset"
    else:
        df = _filtered_activities(tcfg)
        mode = "strict"

    if key == "assay_id" and COL_ASSAY_ID not in df.columns:
        return None, f"{COL_ASSAY_ID}", {"mode": mode, "gate_passed": None}

    smi = pd.Series(CU.canonical_smiles(df[CU.COL_SMILES].astype(str).tolist()),
                    index=df.index)
    df = df[smi.notna()]
    smi = smi[smi.notna()]

    want = set(dataset["smiles"].astype(str))      
    got = set(smi)
    
    
    
    missing, extra = want - got, got - want
    gate = {"mode": mode, "gate_passed": (len(missing) == 0),
            "n_missing": len(missing), "n_extra_dropped": len(extra),
            "missing_sample": sorted(missing)[:3]}

    k = (df[COL_ASSAY_ID].astype(str) if key == "assay_id"
         else df[CU.COL_ASSAY_DESC].astype(str) + " ||| " + df[COL_DOC].astype(str))
    keep = smi.isin(want)
    m = (pd.DataFrame({"smi": smi[keep], "k": k[keep]})
         .groupby("smi")["k"].apply(set).to_dict())
    return m, key, gate



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan


def slice_one_target(slug: str, amap: dict, *, strict_variant: bool):
    """Documentation."""
    pdir = target_results_dir(slug, "p03_protocol_matrix", create=False)
    pred = pd.read_csv(pdir / "predictions.csv")
    pred = pred[pred.model.isin(POOL_TREES)]
    ds = pd.read_csv(ROOT / "data" / slug / "dataset.csv")
    id2 = dict(zip(ds["mol_id"], ds["smiles"].astype(str)))   
    yr = dict(zip(ds["mol_id"], pd.to_numeric(ds["year"], errors="coerce")))

    rows, dropped = [], []
    temp = pred[pred.protocol == TEMPORAL]
    for fold, gf in temp.groupby("fold"):
        cut = int(str(fold).lstrip("T"))
        train_mols = [m for m, y in yr.items() if np.isfinite(y) and y < cut]
        train_assays = set().union(*(amap.get(id2.get(m, ""), set()) for m in train_mols))\
            if train_mols else set()

        for model, g in gf.groupby("model"):
            a = g["mol_id"].map(lambda m: amap.get(id2.get(m, ""), set()))
            if strict_variant:
                seen = a.map(lambda s: bool(s) and s <= train_assays)
            else:
                seen = a.map(lambda s: bool(s & train_assays))
            gs, gn = g[seen.to_numpy()], g[~seen.to_numpy()]
            if len(gs) < MIN_SEEN_N:
                dropped.append({"target": slug, "fold": fold, "model": model,
                                "n_seen": len(gs), "reason": f"n_seen<{MIN_SEEN_N}"})
                continue
            rows.append({
                "target": slug, "fold": fold, "model": model,
                "n_all": len(g), "n_seen": len(gs), "n_novel": len(gn),
                "frac_seen": len(gs) / len(g),
                "rmse_temp_all": rmse(g.y_true.to_numpy(), g.y_pred.to_numpy()),
                "rmse_temp_seen": rmse(gs.y_true.to_numpy(), gs.y_pred.to_numpy()),
                "rmse_temp_novel": (rmse(gn.y_true.to_numpy(), gn.y_pred.to_numpy())
                                    if len(gn) >= 3 else np.nan),
                
                "ysd_all": float(g.y_true.std(ddof=1)),
                "ysd_seen": float(gs.y_true.std(ddof=1)),
                "ysd_novel": float(gn.y_true.std(ddof=1)) if len(gn) > 1 else np.nan,
            })
    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        return per_fold, None, pd.DataFrame(dropped)

    
    sc = pred[pred.protocol == SCAFFOLD]
    sc_fold = (sc.groupby(["model", "fold"])
               .apply(lambda g: rmse(g.y_true.to_numpy(), g.y_pred.to_numpy()),
                      include_groups=False)
               .rename("rmse").reset_index())
    rmse_scaf = float(sc_fold.groupby("model")["rmse"].mean().mean())

    
    fm = per_fold.groupby("model")[
        ["rmse_temp_all", "rmse_temp_seen", "rmse_temp_novel",
         "frac_seen", "ysd_all", "ysd_seen", "ysd_novel"]].mean().mean()
    summ = {
        "target": slug,
        "rmse_scaf": rmse_scaf,
        "rmse_temp_all": fm["rmse_temp_all"],
        "rmse_temp_seen": fm["rmse_temp_seen"],
        "rmse_temp_novel": fm["rmse_temp_novel"],
        "frac_seen": fm["frac_seen"],
        "ysd_all": fm["ysd_all"], "ysd_seen": fm["ysd_seen"], "ysd_novel": fm["ysd_novel"],
        "B_all": fm["rmse_temp_all"] - rmse_scaf,
        "B_seen": fm["rmse_temp_seen"] - rmse_scaf,
        "n_folds_used": per_fold["fold"].nunique(),
        "n_cells_dropped": len(dropped),
    }
    return per_fold, summ, pd.DataFrame(dropped)



def _boot_ci(vals: np.ndarray) -> tuple[float, float, float, float]:
    """Documentation."""
    rng = np.random.default_rng(BOOT_SEED)
    v = vals[np.isfinite(vals)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan, np.nan
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    mid = float(v.mean())
    half = max(mid - lo, hi - mid) * CONSERVATIVE
    return float(lo), float(hi), mid - half, mid + half


def verdict(summ: pd.DataFrame) -> dict:
    """Documentation."""
    n = len(summ)
    k_pos = int((summ.B_seen > 0).sum())
    m_seen, m_all = float(summ.B_seen.mean()), float(summ.B_all.mean())
    r = m_seen / m_all if m_all else np.nan
    lo, hi, clo, chi = _boot_ci(summ.B_seen.to_numpy())
    p = wilcoxon(summ.B_seen, zero_method="zsplit").pvalue if n >= 6 else np.nan

    if k_pos >= 16 and r >= 0.60 and clo > 0:
        v = "message"
    elif (0.30 <= r < 0.60) or (12 <= k_pos <= 15):
        v = "assay"
    else:
        v = "v3"
    return {"n_targets": n, "k_B_seen_pos": k_pos, "mean_B_seen": m_seen,
            "mean_B_all": m_all, "ratio_r": r, "ci_lo": lo, "ci_hi": hi,
            "ci_lo_conservative": clo, "ci_hi_conservative": chi,
            "wilcoxon_p": float(p) if np.isfinite(p) else None, "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()
    slugs = sorted(args.targets) if args.targets else list(FROZEN20)

    frozen = set(slugs) == set(FROZEN20)
    out = (ROOT / "results" / "cross_target" / "e18_assay_drift" if frozen else
           ROOT / "results" / "cross_target" / f"e18_assay_drift__offpool_n{len(slugs)}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[e18] : {DECLARED_IN}")
    print(f"[e18] {len(slugs)} {out.relative_to(ROOT)}"
          + ("" if frozen else "( main table)"))

    per_fold, summ, gates, dropped, failed = [], [], [], [], []
    for s in slugs:
        try:
            ds = pd.read_csv(ROOT / "data" / s / "dataset.csv")
            amap, mode, gate = assay_map(s, ds)
            gate.update({"target": s, "key": mode,
                         "n_mols_mapped": len(amap),
                         "n_assays": len(set().union(*amap.values())) if amap else 0})
            gates.append(gate)
            if gate["mode"] == "strict" and not gate["gate_passed"]:
                failed.append(f"{s}(: dataset {gate['n_missing']}"
                              f"moleculesmissing)")
                continue
            pf, sm, dr = slice_one_target(s, amap, strict_variant=False)
            if sm is None:
                failed.append(f"{s}(available)")
                continue
            pf2, sm2, _ = slice_one_target(s, amap, strict_variant=True)
            sm["B_seen_strict"] = sm2["B_seen"] if sm2 else np.nan
            sm["frac_seen_strict"] = sm2["frac_seen"] if sm2 else np.nan
            per_fold.append(pf), summ.append(sm)
            if len(dr):
                dropped.append(dr)
            print(f"  {s:8s} frac_seen={sm['frac_seen']:.3f} "
                  f"B_all={sm['B_all']:+.4f} B_seen={sm['B_seen']:+.4f}")
        except Exception as e:                       
            failed.append(f"{s}({type(e).__name__}: {e})")

    if failed:
        print(f"[e18] !! target")
        for f in failed:
            print(f"       - {f}")

    G = pd.DataFrame(gates)
    save_table(G, out / "e18_assay_gate.csv")
    if not summ:
        print("[e18] !! target")
        return 1

    S = pd.DataFrame(summ)
    save_table(S, out / "e18_summary.csv")
    save_table(pd.concat(per_fold, ignore_index=True), out / "e18_per_fold.csv")
    if dropped:
        save_table(pd.concat(dropped, ignore_index=True), out / "e18_dropped_cells.csv")

    v = verdict(S)
    print("\n=== §3.6 ===")
    for k in ("n_targets", "k_B_seen_pos", "mean_B_all", "mean_B_seen", "ratio_r",
              "ci_lo_conservative", "ci_hi_conservative"):
        val = v[k]
        print(f"  {k:22s} {val if val is None else round(val, 4)}")
    
    p = v["wilcoxon_p"]
    print(f"  {'wilcoxon_p':22s} {'n/a' if p is None else f'{p:.3e}'}")
    print(f"{'':22s} {v['verdict']}")
    print(f"\n mean(B_seen_strict) = {S.B_seen_strict.mean():.4f}"
          f"frac_seen_strict mean = {S.frac_seen_strict.mean():.3f}")

    
    if "pparg" in slugs:
        try:
            ds = pd.read_csv(ROOT / "data" / "pparg" / "dataset.csv")
            am2, mode2, _ = assay_map("pparg", ds, key="assay_id")
            if am2:
                _, s2, _ = slice_one_target("pparg", am2, strict_variant=False)
                p_proxy = S.set_index("target").loc["pparg"]
                print(f"\n pparg B_seen={p_proxy.B_seen:+.4f}"
                      f"frac_seen={p_proxy.frac_seen:.3f}")
                print(f"pparg assay_id B_seen={s2['B_seen']:+.4f}"
                      f"frac_seen={s2['frac_seen']:.3f}")
                save_table(pd.DataFrame([
                    {"key": "proxy", "B_seen": p_proxy.B_seen, "frac_seen": p_proxy.frac_seen},
                    {"key": "assay_chembl_id", "B_seen": s2["B_seen"],
                     "frac_seen": s2["frac_seen"]},
                ]), out / "e18_pparg_key_sensitivity.csv")
        except Exception as e:
            print(f"[e18] pparg failed {type(e).__name__}: {e}")

    exp_md5 = {}
    for s in slugs:
        try:
            p = Path(load_target(s).chembl_export)
            exp_md5[s] = hashlib.md5(p.read_bytes()).hexdigest()[:12]
        except Exception:
            exp_md5[s] = None

    write_run_meta(out, target="cross_target", stage="e18_assay_drift",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN, "frozen_pool": frozen,
                          "targets": slugs, "failed_targets": failed,
                          "pool_trees": POOL_TREES, "min_seen_n": MIN_SEEN_N,
                          "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                          "conservative_factor": CONSERVATIVE,
                          "superset_slugs": sorted(SUPERSET_SLUGS),
                          "export_md5": exp_md5, "verdict": v})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
