#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import RESULTS_DIR, save_table, write_run_meta  # noqa: E402

from e17_r2_axis import POOL_TREES, axis_gaps, r2_fold_mean, r2_score  # noqa: E402

DEF_E23 = RESULTS_DIR / "cross_target" / "e23_year_missing_sensitivity"
METRICS = [("rmse", "RMSE"), ("r2_fixed", "R²"),
           ("r2", "R²"), ("spearman", "Spearman")]
TOL = 5e-4  


def foldmean_from_pred(pred: pd.DataFrame, slug: str) -> pd.DataFrame:
    """Documentation."""
    var_all = float(pred.drop_duplicates("mol_id")["y_true"].var(ddof=1))
    if not np.isfinite(var_all) or var_all <= 0:
        raise SystemExit(f"[e23b] {slug} y ({var_all})")
    rows = []
    for (protocol, model, fold), g in pred.groupby(["protocol", "model", "fold"]):
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        n = len(yt)
        sse = float(np.sum((yt - yp) ** 2))
        rows.append({
            "protocol": protocol, "model": model, "fold": fold, "n": n,
            "y_sd": float(np.std(yt, ddof=1)) if n > 1 else np.nan,
            "r2": r2_score(yt, yp),
            "r2_fixed": 1.0 - sse / (n * var_all) if n else np.nan,
            "rmse": float(np.sqrt(sse / n)) if n else np.nan,
            "spearman": (float(spearmanr(yt, yp).statistic)
                         if n >= 3 and not np.all(yp == yp[0]) else np.nan),
        })
    pf = pd.DataFrame(rows)
    cols = ["y_sd", "r2", "r2_fixed", "rmse", "spearman"]
    agg = pf.groupby(["protocol", "model"])[cols].mean().reset_index()
    agg["target"] = slug
    agg["var_all"] = var_all
    return agg


def frozen20() -> list[str]:
    man = ROOT / "configs" / "manifests" / "targets_manifest.csv"
    rows = [ln.split(",")[0].strip() for ln in
            man.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    slugs = [s for s in rows[1:] if s]
    if rows[0].split(",")[0].strip() != "slug":
        raise SystemExit(f"[e23b] slug {rows[0]!r}")
    if "pparg" not in slugs:
        slugs.append("pparg")
    if len(slugs) != 20:
        raise SystemExit(f"[e23b] 20 {len(slugs)}")
    return sorted(slugs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e23-dir", default=str(DEF_E23),
                    help="e23 --all-missing __all8")
    ap.add_argument("--tag", default="", help="output results")
    args = ap.parse_args()
    e23_dir = Path(args.e23_dir)
    out = RESULTS_DIR / "cross_target" / ("e23b_crosstarget_substitution"
                                         + (f"__{args.tag}" if args.tag else ""))
    prot = load_protocol()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[e23b] {e23_dir.name} {out.name}")
    print(f"[e23b] {POOL_TREES} e17")

    pred23 = pd.read_csv(e23_dir / "e23_predictions.csv")
    aligned = sorted(pred23.target.unique())
    print(f"[e23b] e23 {aligned}")

    fm_frozen = pd.concat([r2_fold_mean(s)[0] for s in frozen20()],
                          ignore_index=True)
    gaps_frozen = axis_gaps(fm_frozen)

    fm_aligned = pd.concat([foldmean_from_pred(pred23[pred23.target == s], s)
                            for s in aligned], ignore_index=True)
    gaps_aligned = axis_gaps(fm_aligned)

    
    checks = []
    for metric in ("rmse", "spearman", "r2"):
        a = fm_aligned[fm_aligned.protocol == "temporal"].set_index(
            ["target", "model"])[metric]
        f = fm_frozen[(fm_frozen.protocol == "temporal")
                      & (fm_frozen.target.isin(aligned))].set_index(
            ["target", "model"])[metric]
        dev = (a - f.reindex(a.index)).abs().max()
        ok = dev <= TOL
        print(f"[] temporal {metric:9s} {dev:.6f}"
              f"→ {'✅' if ok else '❌'}")
        checks.append({"metric": metric, "temporal_max_abs_dev": dev, "ok": ok})
        if not ok:
            raise SystemExit(f"[e23b] temporal {metric} {dev:.6f}"
                             "temporal reason")
    

    rows, summ = [], []
    for tag, lab in METRICS:
        gf = gaps_frozen.set_index("target")
        ga = gaps_aligned.set_index("target")
        A = gf[f"A_{tag}"].copy()
        B = gf[f"B_{tag}"].copy()
        A_used, B_used = A.copy(), B.copy()
        for s in aligned:
            A_used[s], B_used[s] = ga.loc[s, f"A_{tag}"], ga.loc[s, f"B_{tag}"]
        ratio_f, ratio_u = B.mean() / A.mean(), B_used.mean() / A_used.mean()
        summ.append({"metric": tag, "label": lab,
                     "A_mean_frozen": A.mean(), "B_mean_frozen": B.mean(),
                     "A_mean_substituted": A_used.mean(),
                     "B_mean_substituted": B_used.mean(),
                     "B_over_A_frozen": ratio_f,
                     "B_over_A_substituted": ratio_u,
                     "relative_shift": ratio_u / ratio_f - 1,
                     "n_B_positive_frozen": int((B > 0).sum()),
                     "n_B_positive_substituted": int((B_used > 0).sum()),
                     "n_targets": len(A), "n_substituted": len(aligned)})
        for s in A.index:
            rows.append({"target": s, "metric": tag, "substituted": s in aligned,
                         "A_frozen": A[s], "B_frozen": B[s],
                         "A_used": A_used[s], "B_used": B_used[s]})

    summary = pd.DataFrame(summ)
    print(f"\n{'':<16} {' B/A':>9} {'':>8} {'':>9}"
          f"{'B>0 ':>9} {'B>0 ':>9}")
    print("-" * 68)
    for r in summ:
        print(f"{r['label']:<16} {r['B_over_A_frozen']:>9.3f} "
              f"{r['B_over_A_substituted']:>8.3f} {r['relative_shift']:>+9.2%} "
              f"{r['n_B_positive_frozen']:>7}/20 "
              f"{r['n_B_positive_substituted']:>7}/20")

    save_table(pd.DataFrame(rows), out / "e23b_per_target_gaps.csv",
               float_fmt="%.8f")
    save_table(summary, out / "e23b_crosstarget_shift.csv", float_fmt="%.8f")
    save_table(pd.DataFrame(checks), out / "e23b_temporal_identity_check.csv",
               float_fmt="%.8f")
    write_run_meta(out, target="__cross_target__",
                   stage="e23b_crosstarget_substitution",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"substituted_targets": aligned,
                          "axis_definition": "e17: A = random − scaffold, "
                                             "B = scaffold − temporal",
                          "pool": POOL_TREES, "pool_aggregation": "mean",
                          "cross_target_aggregation": "mean over 20 targets "
                                                      "(matches e17 / 57)",
                          "r2_fixed_denominator_aligned_arm": "subset variance "
                              "(year-present molecules only)",
                          "not_done": ([] if len(aligned) == 8 else
                                      [f"{len(aligned)}/8 missing yeartarget"
                                       "message"]),
                          "notes": "note_8/31 §2.5 57 2.88×"
                                   "missing yearmoleculestemporal"
                                   "⛔ output 2.88×"})
    print(f"\n[e23b] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
