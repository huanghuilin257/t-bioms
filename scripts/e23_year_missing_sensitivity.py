#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from pparbench.threads import apply_thread_limits  # noqa: E402

N_THREADS = apply_thread_limits()  

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

from pparbench.config import load_protocol, load_target  # noqa: E402
from pparbench.io import (RESULTS_DIR, features_path, save_table,  # noqa: E402
                          write_run_meta)
from pparbench.metrics import compute_metrics  # noqa: E402
from pparbench.models import build_models, family_of  # noqa: E402

from p03_protocol_matrix import build_folds  # noqa: E402

PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]
TREES4 = ["lightgbm", "hgb", "rf", "extratrees"]  

DEFAULT_TARGETS = ["esr1", "nr3c1"]

SKIPPED = {"oprm1": -0.0019, "egfr": -0.0014, "drd2": -0.0013,
           "htr2a": +0.0013, "ache": +0.0006, "kcnh2": +0.0004}

ALL_MISSING = sorted(DEFAULT_TARGETS + list(SKIPPED))


def run_subset(slug: str, prot, tcfg, keep: np.ndarray, X, y, year, scaffold,
               mol_id):
    """Documentation."""
    Xs, ys = X[keep], y[keep]
    years, scafs, mids = year[keep], scaffold[keep], mol_id[keep]
    models = build_models(prot.models, seed=prot.seed, n_jobs=prot.n_jobs,
                          boosting_n_jobs=prot.omp_num_threads,
                          fp_bits=int(prot.features["morgan_bits"]))
    enrich = prot.metrics.get("enrichment_pct", [])
    rows, preds = [], []
    for protocol in PROTOCOLS:
        folds = build_folds(protocol, prot=prot, tcfg=tcfg, year=years,
                            scaffold=scafs, n=len(ys))
        if not folds:
            print(f"[warn] {protocol} valid fold skip")
            continue
        for mname, proto in models.items():
            t0 = time.time()
            per_fold = []
            for fold_name, tr, te in folds:
                mdl = clone(proto)
                mdl.fit(Xs[tr], ys[tr])
                pred = np.asarray(mdl.predict(Xs[te]), dtype=float)
                met = compute_metrics(ys[te], pred, enrichment_pct=enrich)
                per_fold.append({"fold": fold_name, "n_train": len(tr),
                                 "n": len(te), **met})
                preds.append(pd.DataFrame({
                    "target": slug, "protocol": protocol, "model": mname,
                    "fold": fold_name, "mol_id": mids[te],
                    "y_true": ys[te], "y_pred": pred}))
            df = pd.DataFrame(per_fold)
            rows.append({"protocol": protocol, "model": mname,
                         "family": family_of(mname), "n_folds": len(df),
                         **{c: df[c].mean() for c in df.columns
                            if c not in ("fold",)}})
            print(f"  {protocol:16s} {mname:11s} "
                  f"spearman={rows[-1]['spearman']:.4f}  {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def frozen_foldmean(slug: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS_DIR / slug / "p03_protocol_matrix" / "per_fold.csv")
    return (d.groupby(["protocol", "model"])["spearman"].mean()
            .reset_index().rename(columns={"spearman": "spearman_frozen"}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--all-missing", action="store_true",
                    help="8 missing yearmoleculestarget --targets"
                         "e23b")
    ap.add_argument("--tag", default="",
                    help="output results")
    ap.add_argument("--tol", type=float, default=5e-4,
                    help="temporal default 5e-4 %%.4f")
    args = ap.parse_args()
    targets = ALL_MISSING if args.all_missing else args.targets

    prot = load_protocol()
    name = "e23_year_missing_sensitivity" + (f"__{args.tag}" if args.tag else "")
    out = RESULTS_DIR / "cross_target" / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"[e23] {len(targets)}" + " ".join(targets))
    skipped_now = {t: v for t, v in SKIPPED.items() if t not in targets}
    if skipped_now:
        print("[e23] y_sd ≤0.19%")
        for t, v in sorted(skipped_now.items(), key=lambda kv: kv[1]):
            print(f"       {t:8s} narrowing={v:+.4%}")
    else:
        print("[e23] 8 missing yeartarget SKIPPED")

    all_rows, all_preds, checks = [], [], []
    for slug in targets:
        tcfg = load_target(slug)
        d = np.load(features_path(slug, "ligand2057"), allow_pickle=True)
        X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
        year, scaffold, mol_id = d["year"], d["scaffold"], d["mol_id"]
        keep = ~pd.isna(pd.Series(year)).to_numpy()
        print(f"\n=== {slug} n={len(y)} year n={keep.sum()}"
              f"{len(y)-keep.sum()} ===")
        if keep.all():
            print("missing yearmolecules skip")
            continue

        sub, pred = run_subset(slug, prot, tcfg, keep, X, y, year, scaffold,
                              mol_id)
        sub["target"] = slug
        sub["arm"] = "year_present_only"
        all_rows.append(sub)
        all_preds.append(pred)

        
        fz = frozen_foldmean(slug)
        mg = sub.merge(fz, on=["protocol", "model"])
        t = mg[mg.protocol == "temporal"].copy()
        t["abs_dev"] = (t.spearman - t.spearman_frozen).abs()
        worst = t.abs_dev.max()
        print(f"[] temporal {worst:.6f} {args.tol}"
              f"{'✅ ' if worst <= args.tol else '❌ '}")
        if worst > args.tol:
            print(t[["model", "spearman", "spearman_frozen", "abs_dev"]]
                  .round(6).to_string(index=False))
            raise SystemExit(f"[e23] {slug} temporal"
                             "temporal reason")
        checks.append({"target": slug, "temporal_max_abs_dev": worst,
                       "n_all": len(y), "n_year_present": int(keep.sum())})

        
        
        
        
        
        
        for pool, name in [(TREES4, "trees4"), (None, "all7")]:
            m = mg if pool is None else mg[mg.model.isin(pool)]
            p_new = m.groupby("protocol")["spearman"].mean()
            p_old = m.groupby("protocol")["spearman_frozen"].mean()
            for label, ser in [("CV missing yearmolecules", p_old),
                               ("CV", p_new)]:
                A = ser["random_holdout"] - ser["scaffold_cv"]
                B = ser["scaffold_cv"] - ser["temporal"]
                ratio = B / A if A else float("nan")
                print(f"  [{name:6s}] {label:24s} "
                      f"A={A:+.4f}  B={B:+.4f}  B/A={ratio:.3f}")
            all_rows.append(pd.DataFrame([{
                "target": slug, "arm": "comparison", "pool": name,
                "metric": "spearman", "aggregation": "tree-pool mean, e17 axis",
                "A_frozen": p_old["random_holdout"] - p_old["scaffold_cv"],
                "B_frozen": p_old["scaffold_cv"] - p_old["temporal"],
                "A_aligned": p_new["random_holdout"] - p_new["scaffold_cv"],
                "B_aligned": p_new["scaffold_cv"] - p_new["temporal"],
            }]))

    res = pd.concat(all_rows, ignore_index=True)
    
    save_table(res, out / "e23_foldmean_and_gaps.csv", float_fmt="%.8f")
    save_table(pd.concat(all_preds, ignore_index=True),
               out / "e23_predictions.csv", float_fmt="%.8f")
    save_table(pd.DataFrame(checks), out / "e23_temporal_identity_check.csv",
               float_fmt="%.8f")

    write_run_meta(out, target="__cross_target__",
                   stage="e23_year_missing_sensitivity",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"targets": targets, "skipped": skipped_now,
                          "all_missing_year_targets": ALL_MISSING,
                          "protocols": PROTOCOLS, "tol": args.tol,
                          "notes": "note_8/31 §2.4 missing yearmolecules CV A"
                                   "B molecules"
                                   "temporal"
                                   + ("8 missing yeartarget SKIPPED"
                                      if not skipped_now else
                                      "y_sd ≥0.5% skipped")})
    print(f"\n[e23] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
