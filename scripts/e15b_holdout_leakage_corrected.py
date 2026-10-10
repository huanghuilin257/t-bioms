#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits  # noqa: E402

apply_thread_limits()  

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import save_table, write_run_meta  # noqa: E402

E15_DIR = ROOT / "results" / "cross_target" / "e15_landrum_preflight"
OUT_DIR = ROOT / "results" / "cross_target" / "e15b_leakage_corrected"

DELTA_CSV = E15_DIR / "e15_delta_by_target.csv"
POWER_CSV = E15_DIR / "e15_power.csv"


OVERLAP = {"drd3": 0.8547, "htr1a": 0.4296, "htr6": 0.2970,
           "slc6a3": 0.2065, "adra1d": 0.2045, "hrh4": 0.0321,
           "trpv1": 0.0207, "mtnr1a": 0.0011}
HARD_EXCLUDED = ["drd3", "htr1a"]          
FLAGGED = ["htr6", "slc6a3", "adra1d"]     
CLEAN = ["hrh4", "trpv1", "mtnr1a"]        
OVERLAP_THRESHOLD = 0.30

MAIN6 = FLAGGED + CLEAN                    
CLEAN3 = CLEAN                             


PARTIAL_CUTPOINTS = {"mtnr1a": 4, "trpv1": 2}

CRIT_GO = 47                               
HEADLINE_POOL, HEADLINE_METRIC = "trees4", "spearman"


def power_row(delta: np.ndarray, label: str) -> dict:
    """Documentation."""
    d = np.asarray([x for x in delta if np.isfinite(x)], dtype=float)
    n = len(d)
    out = {"subset": label, "n_targets": n,
           "n_zero_delta": int(np.sum(np.isclose(d, 0.0, atol=1e-12))),
           "delta_mean": float(np.mean(d)) if n else np.nan,
           "delta_sd": float(np.std(d, ddof=1)) if n > 1 else np.nan,
           "delta_min": float(np.min(d)) if n else np.nan,
           "delta_max": float(np.max(d)) if n else np.nan,
           "n_positive": int(np.sum(d > 1e-12)), "n_negative": int(np.sum(d < -1e-12))}
    sd, mean = out["delta_sd"], out["delta_mean"]
    if n > 1 and np.isfinite(sd) and np.isfinite(mean) and abs(mean) > 1e-12:
        out["t_needed"] = int(math.ceil((1.96 * sd / abs(mean)) ** 2))
    else:
        out["t_needed"] = np.nan
    out["sd_rel_se"] = float(1.0 / math.sqrt(2 * (n - 1))) if n > 1 else np.nan
    return out


def selfcalibrate(dl: pd.DataFrame) -> bool:
    """Documentation."""
    landed = pd.read_csv(POWER_CSV)
    ok = True
    for (pool, metric), g in dl.groupby(["pool", "metric"]):
        for label, sub in (("full7_n6", g[g["full7"]]), ("all_usable_n8", g)):
            mine = power_row(sub["delta"].to_numpy(), label)
            ref = landed[(landed["pool"] == pool) & (landed["metric"] == metric)
                         & (landed["subset"] == label)]
            if len(ref) != 1:
                print(f"[gate] missing {pool}/{metric}/{label}", file=sys.stderr)
                ok = False
                continue
            ref = ref.iloc[0]
            for col in ("n_targets", "n_zero_delta", "n_positive", "n_negative"):
                if int(mine[col]) != int(ref[col]):
                    print(f"[gate] {pool}/{metric}/{label} {col}: "
                          f"{mine[col]} ≠ {ref[col]}", file=sys.stderr)
                    ok = False
            for col in ("delta_mean", "delta_sd", "delta_min", "delta_max", "sd_rel_se"):
                a, b = float(mine[col]), float(ref[col])
                if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=1e-4, atol=1e-9):
                    print(f"[gate] {pool}/{metric}/{label} {col}: {a!r} ≠ {b!r}"
                          f"input %.6g 6 valid", file=sys.stderr)
                    ok = False
            a, b = mine["t_needed"], ref["t_needed"]
            if not (pd.isna(a) and pd.isna(b)) and int(a) != int(b):
                print(f"[gate] {pool}/{metric}/{label} t_needed: {a} ≠ {b}",
                      file=sys.stderr)
                ok = False
    return ok


def verdict_for(tn: float, n: int, label: str) -> str:
    if n < 2:
        return (f"{label} {n} target sd(Δ) T_needed"
                f"hrh4 note_8/24 §1.4")
    if not np.isfinite(tn):
        return ("Δ̄≈0 T_needed"
                "no")
    if tn <= CRIT_GO:
        return f"1 T_needed={tn:.0f} ≲ {CRIT_GO} H1"
    return (f"3 T_needed={tn:.0f} > {CRIT_GO} ****"
            f"results")


def main() -> int:
    if not DELTA_CSV.exists() or not POWER_CSV.exists():
        print(f"[e15b] e15 run first scripts/e15_landrum_preflight.py", file=sys.stderr)
        return 1

    dl = pd.read_csv(DELTA_CSV)

    print("[e15b] Δ e15")
    if not selfcalibrate(dl):
        print("[e15b] failed", file=sys.stderr)
        return 1
    print("[e15b] 16 ×metrics× \n")

    present = set(dl["target"])
    subsets = {
        "e15_full7_n6_": sorted(set(dl[dl["full7"]]["target"])),
        "e15_all_usable_n8_": sorted(present),
        "main6_100": sorted(set(MAIN6) & present),
        "clean3_100": sorted(set(CLEAN3) & present),
        "main6_7": sorted((set(MAIN6) & present) - set(PARTIAL_CUTPOINTS)),
        "clean3_7": sorted((set(CLEAN3) & present) - set(PARTIAL_CUTPOINTS)),
    }
    for k, v in subsets.items():
        print(f"  {k:26s} n={len(v)}  {';'.join(v)}")
    print()

    rows, tails, vsrand = [], [], []
    for (pool, metric), g in dl.groupby(["pool", "metric"]):
        g = g.set_index("target")
        for label, tg in subsets.items():
            sub = g.loc[[t for t in tg if t in g.index]]
            r = power_row(sub["delta"].to_numpy(), label)
            r.update({"pool": pool, "metric": metric,
                      "targets": ";".join(sub.index),
                      "zero_by_same_model": ";".join(sub[sub["same_model"]].index)})
            rows.append(r)
            
            if len(sub):
                tails.append({
                    "pool": pool, "metric": metric, "subset": label, "n_targets": len(sub),
                    "max_regret_prescribed": float(sub["regret_prescribed"].max()),
                    "max_regret_cv_selfpick": float(sub["regret_cv_selfpick"].max()),
                    "argmax_prescribed": str(sub["regret_prescribed"].idxmax()),
                    "argmax_cv_selfpick": str(sub["regret_cv_selfpick"].idxmax()),
                    "mean_random_pick_regret": float(sub["random_pick_regret"].mean()),
                    "n_zero_regret_prescribed": int((sub["regret_prescribed"] <= 1e-12).sum()),
                    "n_zero_regret_cv_selfpick": int((sub["regret_cv_selfpick"] <= 1e-12).sum()),
                })
                
                
                
                d_r = (sub["random_pick_regret"] - sub["regret_prescribed"]).to_numpy()
                vsrand.append({
                    "pool": pool, "metric": metric, "subset": label, "n_targets": len(sub),
                    "mean_regret_prescribed": float(sub["regret_prescribed"].mean()),
                    "mean_regret_random_pick": float(sub["random_pick_regret"].mean()),
                    "delta_vs_random_mean": float(np.mean(d_r)),
                    "n_prescribed_better": int(np.sum(d_r > 1e-12)),
                    "n_prescribed_worse": int(np.sum(d_r < -1e-12)),
                    "prescribed_worse_on": ";".join(sub.index[d_r < -1e-12]),
                })

    pw = pd.DataFrame(rows)[[
        "pool", "metric", "subset", "n_targets", "n_zero_delta", "delta_mean", "delta_sd",
        "delta_min", "delta_max", "n_positive", "n_negative", "t_needed", "sd_rel_se",
        "targets", "zero_by_same_model"]].sort_values(["pool", "metric", "subset"])
    th = pd.DataFrame(tails).sort_values(["pool", "metric", "subset"])
    vr = pd.DataFrame(vsrand).sort_values(["pool", "metric", "subset"])

    vd = []
    for label in subsets:
        r = pw[(pw["pool"] == HEADLINE_POOL) & (pw["metric"] == HEADLINE_METRIC)
               & (pw["subset"] == label)]
        if not len(r):
            continue
        r = r.iloc[0]
        tn = float(r["t_needed"]) if pd.notna(r["t_needed"]) else np.nan
        vd.append({"subset": label, "pool": HEADLINE_POOL, "metric": HEADLINE_METRIC,
                   "n_targets": int(r["n_targets"]), "delta_mean": float(r["delta_mean"]),
                   "t_needed": tn, "verdict": verdict_for(tn, int(r["n_targets"]), label)})
    vd = pd.DataFrame(vd)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_table(pw, OUT_DIR / "e15b_power_by_subset.csv", float_fmt="%.6g")
    save_table(th, OUT_DIR / "e15b_tail_posthoc.csv", float_fmt="%.6g")
    save_table(vr, OUT_DIR / "e15b_vs_randompick_posthoc.csv", float_fmt="%.6g")
    save_table(vd, OUT_DIR / "e15b_verdict.csv", float_fmt="%.6g")

    hp, hm = HEADLINE_POOL, HEADLINE_METRIC
    print(f"[e15b] {hp} × {hm} 100")
    show = pw[(pw["pool"] == hp) & (pw["metric"] == hm)][
        ["subset", "n_targets", "n_zero_delta", "delta_mean", "delta_sd",
         "n_positive", "n_negative", "t_needed", "sd_rel_se"]]
    print(show.to_string(index=False))
    print("\n[e15b]")
    for _, r in vd.iterrows():
        print(f"  {r['subset']:26s} {r['verdict']}")
    print(f"\n[e15b] 16 ×metrics × {len(subsets)} {OUT_DIR}")

    print("\n[e15b] post-hoc vs H2 ×metrics 88 p")
    print(vr[(vr["pool"] == hp) & (vr["metric"] == hm)][
        ["subset", "n_targets", "mean_regret_prescribed", "mean_regret_random_pick",
         "delta_vs_random_mean", "n_prescribed_better", "n_prescribed_worse",
         "prescribed_worse_on"]].to_string(index=False))

    write_run_meta(OUT_DIR, target="cross_target", stage="e15b_leakage_corrected",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={
                       "declared_in": "manuscript leakage-control declaration",
                       "why": "e15 full7_n6 drd3/htr1a"
                              "6 target",
                       "no_refit": "Δ_j FROZEN20",
                       "input_tables": [str(DELTA_CSV), str(POWER_CSV)],
                       "input_precision": "%.6g 6 valid",
                       "overlap": OVERLAP, "overlap_threshold": OVERLAP_THRESHOLD,
                       "hard_excluded": HARD_EXCLUDED, "flagged": FLAGGED, "clean": CLEAN,
                       "partial_cutpoints": PARTIAL_CUTPOINTS,
                       "subsets": {k: v for k, v in subsets.items()},
                       "crit_go": CRIT_GO,
                       "headline": {"pool": hp, "metric": hm, "subset": "main6_100"},
                       "posthoc_tables": ["e15b_tail_posthoc.csv",
                                          "e15b_vs_randompick_posthoc.csv"],
                       "posthoc_disclosure": "vs-random H1"
                                             "post-hoc 88"
                                             "mean p",
                   })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
