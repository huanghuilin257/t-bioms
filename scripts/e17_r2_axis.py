#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()          

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402
from scipy.stats import wilcoxon, spearmanr    # noqa: E402

from pparbench.io import target_results_dir, save_table, write_run_meta  # noqa: E402
from pparbench.config import load_protocol                               # noqa: E402

DECLARED_IN = "manuscript protocol declaration"

RANDOM = "random_holdout"
SCAFFOLD = "scaffold_cv"
TEMPORAL = "temporal"
POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]     


FROZEN20 = [
    "abl1", "ache", "adora2a", "ar", "bace1", "drd2", "egfr", "esr1", "f10", "f2",
    "hdac1", "htr2a", "kcnh2", "kdr", "nr3c1", "oprm1", "pik3ca", "ppara", "ppard", "pparg",
]



COMPETITOR_GAP_R2 = {"egfr": 0.140, "drd2": 0.084, "bace1": 0.085, "kcnh2": 0.171}

N_BOOT = 2000
BOOT_SEED = 20260825
CONSERVATIVE = 1.222        


# ----------------------------------------------------------------- R²
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Documentation."""
    if len(y_true) < 3:
        return np.nan
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if not np.isfinite(sst) or sst <= 0:
        return np.nan          
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def r2_fold_mean(slug: str) -> pd.DataFrame:
    """Documentation."""
    p = target_results_dir(slug, "p03_protocol_matrix", create=False) / "predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    pred = pd.read_csv(p)

    
    var_all = float(pred.drop_duplicates("mol_id")["y_true"].var(ddof=1))
    if not np.isfinite(var_all) or var_all <= 0:
        raise SystemExit(f"[e17] {slug} data y ({var_all})")

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
    per_fold = pd.DataFrame(rows)
    cols = ["y_sd", "r2", "r2_fixed", "rmse", "spearman"]
    agg = per_fold.groupby(["protocol", "model"])[cols].mean().reset_index()
    agg["n_folds"] = per_fold.groupby(["protocol", "model"])["fold"].nunique().to_numpy()
    agg["n_neg_folds"] = (per_fold.assign(neg=per_fold.r2 < 0)
                          .groupby(["protocol", "model"])["neg"].sum().to_numpy())
    agg["target"] = slug
    agg["var_all"] = var_all
    return agg, per_fold.assign(target=slug)



def axis_gaps(foldmean: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    t = foldmean[foldmean.model.isin(POOL_TREES)]
    out = pd.DataFrame(index=sorted(t.target.unique()))
    out.index.name = "target"
    for metric in ("r2", "r2_fixed", "rmse", "spearman", "y_sd"):
        piv = t.pivot_table(index="target", columns="protocol", values=metric)
        miss = [c for c in (RANDOM, SCAFFOLD, TEMPORAL) if c not in piv.columns]
        if miss:
            raise SystemExit(f"[e17] {metric} protocol {miss}")
        for proto, tag in ((RANDOM, "rand"), (SCAFFOLD, "scaf"), (TEMPORAL, "temp")):
            out[f"{metric}_{tag}"] = piv[proto]
        if metric == "rmse":            
            out["A_rmse"] = piv[SCAFFOLD] - piv[RANDOM]
            out["B_rmse"] = piv[TEMPORAL] - piv[SCAFFOLD]
        elif metric != "y_sd":          
            out[f"A_{metric}"] = piv[RANDOM] - piv[SCAFFOLD]
            out[f"B_{metric}"] = piv[SCAFFOLD] - piv[TEMPORAL]
    
    out["ysd_temp_over_scaf"] = out["y_sd_temp"] / out["y_sd_scaf"]
    out["B_minus_A_rmse"] = out["B_rmse"] - out["A_rmse"]
    return out.reset_index()



def _boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Documentation."""
    est = float(np.mean(x))
    boots = np.array([np.mean(rng.choice(x, size=len(x), replace=True))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    half = (hi - lo) / 2.0 * CONSERVATIVE
    return est, half, est - half, est + half


def direction_test(x: np.ndarray, label: str, rng: np.random.Generator) -> dict:
    """Documentation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    k = int(np.sum(x > 0))
    try:
        w_stat, w_p = wilcoxon(x, zero_method="zsplit", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    est, half, lo, hi = _boot_ci(x, rng)
    return {"claim": label, "n_targets": n, "k_positive": k,
            "mean": est, "sd": float(np.std(x, ddof=1)) if n > 1 else np.nan,
            "median": float(np.median(x)),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "boot_halfwidth_conservative": half,
            "ci_lo_conservative": lo, "ci_hi_conservative": hi}


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="e17 R² protocol")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    if args.targets:
        slugs = sorted(args.targets)
    else:
        slugs = sorted(FROZEN20)
        rd = ROOT / "results"
        found = sorted(d.name for d in rd.iterdir()
                       if d.is_dir() and d.name != "cross_target"
                       and (d / "p03_protocol_matrix" / "predictions.csv").exists())
        extra = sorted(set(found) - set(FROZEN20))
        if extra:
            
            
            print(f"[e17] results/ {len(extra)} target"
                  f"**** {extra}")

    is_frozen = set(slugs) == set(FROZEN20)
    if is_frozen:
        out = ROOT / "results" / "cross_target" / "e17_r2_axis"
    else:
        out = ROOT / "results" / "cross_target" / f"e17_r2_axis__offpool_n{len(slugs)}"
        print(f"[e17] !! target 20 output {out.name}/")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[e17] : {DECLARED_IN}")
    print(f"[e17] target {len(slugs)}: {slugs}")
    print(f"[e17] {POOL_TREES}")

    aggs, per_folds, failed = [], [], []
    for s in slugs:
        try:
            a, pf = r2_fold_mean(s)
        except FileNotFoundError as e:
            failed.append(s)
            print(f"[e17] !! {s} predictions.csv: {e}")
            continue
        aggs.append(a)
        per_folds.append(pf)
    if failed:
        
        print(f"[e17] !! target : {failed}")
        return 1

    foldmean = pd.concat(aggs, ignore_index=True)
    per_fold = pd.concat(per_folds, ignore_index=True)
    save_table(foldmean, out / "e17_r2_foldmean.csv")
    save_table(per_fold, out / "e17_r2_per_fold.csv")

    gaps = axis_gaps(foldmean)
    save_table(gaps, out / "e17_r2_axis_gap.csv")

    
    neg = (per_fold[per_fold.r2 < 0]
           .groupby(["target", "protocol"])
           .agg(n_neg_folds=("r2", "size"), worst_r2=("r2", "min"))
           .reset_index().sort_values("worst_r2"))
    save_table(neg, out / "e17_r2_negative_folds.csv")
    print(f"\n[e17] R²<0 (target,protocol) : {len(neg)}")
    if len(neg):
        print(neg.head(15).to_string(index=False))

    
    rng = np.random.default_rng(BOOT_SEED)
    tests = []
    for tag, lab in (("rmse", "RMSE"), ("r2_fixed", "R²"),
                     ("r2", "R²()"), ("spearman", "Spearman")):
        tests.append(direction_test(gaps[f"B_{tag}"].to_numpy(),
                                   f"R1[{lab}]: B > 0", rng))
        tests.append(direction_test((gaps[f"B_{tag}"] - gaps[f"A_{tag}"]).to_numpy(),
                                   f"R2[{lab}]: B > A", rng))
    tests_df = pd.DataFrame(tests)
    save_table(tests_df, out / "e17_r2_test.csv")

    
    ys = gaps["ysd_temp_over_scaf"].to_numpy()
    rho, pv = spearmanr(ys, gaps["B_r2"].to_numpy())
    rho_f, pv_f = spearmanr(ys, gaps["B_rmse"].to_numpy())
    
    favorable = gaps[gaps.ysd_temp_over_scaf >= 1.0]
    diag = pd.DataFrame([{
        "ysd_ratio_median": float(np.median(ys)),
        "ysd_ratio_min": float(np.min(ys)), "ysd_ratio_max": float(np.max(ys)),
        "n_targets_ysd_shrunk": int((ys < 1).sum()), "n_targets": len(ys),
        "spearman_ysdratio_vs_B_r2": float(rho), "p_ysdratio_vs_B_r2": float(pv),
        "spearman_ysdratio_vs_B_rmse": float(rho_f), "p_ysdratio_vs_B_rmse": float(pv_f),
        "n_favorable": len(favorable),
        "favorable_targets": " ".join(sorted(favorable.target)),
        "favorable_B_r2_mean": float(favorable.B_r2.mean()),
        "favorable_B_rmse_mean": float(favorable.B_rmse.mean()),
        "favorable_B_r2fixed_mean": float(favorable.B_r2_fixed.mean()),
        "all_B_r2_mean": float(gaps.B_r2.mean()),
        "all_B_rmse_mean": float(gaps.B_rmse.mean()),
        "verdict": ("target"
                    "RMSE/Spearman 20/20"),
    }])
    save_table(diag, out / "e17_ysd_confound.csv")

    
    ov = [t for t in COMPETITOR_GAP_R2 if t in set(gaps.target)]
    rep = gaps[gaps.target.isin(ov)][["target", "A_r2", "B_r2", "B_rmse"]].copy()
    rep["competitor_gap_R2"] = rep.target.map(COMPETITOR_GAP_R2)
    rep["ratio_ours_over_theirs"] = rep.A_r2 / rep.competitor_gap_R2
    save_table(rep, out / "e17_r2_vs_competitor.csv")

    
    print("\n=== targetmean ===")
    show = ["target", "B_rmse", "B_r2_fixed", "B_r2", "B_spearman", "ysd_temp_over_scaf"]
    print(gaps[show].round(4).to_string(index=False))
    print("\n=== RMSE ===")
    print(tests_df[["claim", "n_targets", "k_positive", "mean", "sd",
                    "wilcoxon_p", "ci_lo_conservative",
                    "ci_hi_conservative"]].round(4).to_string(index=False))
    print("\n=== ===")
    for tag, lab, immune in (("rmse", "RMSE", "message"),
                             ("r2_fixed", "R²", "message"),
                             ("r2", "R²", "****"),
                             ("spearman", "Spearman", "message")):
        B, A = gaps[f"B_{tag}"], gaps[f"A_{tag}"]
        print(f"{lab:14s} {immune:8s} B mean {B.mean():7.4f}"
              f"B>0 {int((B>0).sum())}/{len(B)} A mean {A.mean():7.4f}"
              f"B/A {B.mean()/A.mean():.2f}x")
    print("\n=== ===")
    print(f"y_sd (temporal/scaffold): {np.median(ys):.3f}"
          f"{ys.min():.3f}~{ys.max():.3f} target {(ys<1).sum()}/{len(ys)}")
    print(f"Spearman(y_sd , B_R2) = {rho:+.3f} p={pv:.4f}")
    print(f"Spearman(y_sd , B_RMSE) = {rho_f:+.3f} p={pv_f:.4f}")
    print(f"{len(favorable)} target y B_RMSE mean"
          f"{favorable.B_rmse.mean():.4f} {gaps.B_rmse.mean():.4f}")
    print(f"    {sorted(favorable.target)}")
    print("\n=== R3 A_r2 vs Table 7 metrics ===")
    print(rep.round(4).to_string(index=False))

    write_run_meta(out, target="__cross_target__", stage="e17_r2_axis",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN, "targets": slugs,
                          "is_frozen_pool": is_frozen, "pool_trees": POOL_TREES,
                          "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                          "conservative_factor": CONSERVATIVE,
                          "primary_metric": "rmse",
                          "metrics": ["rmse", "r2_fixed", "r2", "spearman"],
                          "r2_definition": "1 - SSE/SST, ybar = fold test mean, not truncated",
                          "r2_fixed_definition": "1 - SSE/(n * var(y over whole dataset))",
                          "ysd_confound": ("R² test data y"
                                           "§7 RMSE"),
                          "competitor": "Alshehade et al., JCIM 2026, 66, 8923-8941, Table 7"})
    print(f"\n[e17] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
