#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                             # noqa: E402

from pparbench.config import load_protocol                       # noqa: E402
from pparbench.io import RESULTS_DIR, save_table, write_run_meta # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="e04 target")
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    frames, missing = [], []
    for t in args.targets:
        p = RESULTS_DIR / t / "e04_hparam_pool" / "regret.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            missing.append(t)
    if missing:
        print(f"[warn] {len(missing)} target : {missing}")
    if not frames:
        print("[error] no e04 results")
        return 1
    df = pd.concat(frames, ignore_index=True)
    n_t = df["target"].nunique()

    outdir = RESULTS_DIR / "cross_target" / "e04_hparam_pool"
    outdir.mkdir(parents=True, exist_ok=True)
    save_table(df, outdir / "e04_all_targets.csv")

    # ---------- Q1 ----------
    print(f"\n===== Q1 boosting vs rf {n_t} target target =====")
    q1 = (df.groupby(["metric", "selector"])
            .agg(n_targets=("target", "nunique"),
                 cv_boost_wins=("tuned_boosting_beats_default_rf", "sum"),
                 oracle_boost_wins=("oracle_boosting_beats_default_rf", "sum"),
                 mean_gap_rf_minus_cvboost=("gap_rf_minus_tunedboost", "mean"),
                 mean_rf=("rf_default_temporal", "mean"),
                 mean_cv_boost=("tuned_boosting_temporal", "mean"),
                 mean_oracle_boost=("best_boosting_temporal", "mean"))
            .reset_index())
    print(q1.round(4).to_string(index=False))
    save_table(q1, outdir / "e04_q1_boosting_vs_rf.csv")

    print("\n--- target spearman / random_holdout ---")
    d = df[(df["metric"] == "spearman") & (df["selector"] == "random_holdout")]
    print(d[["target", "tuned_boosting", "tuned_boosting_temporal",
             "rf_default_temporal", "gap_rf_minus_tunedboost",
             "best_boosting_temporal",
             "oracle_boosting_beats_default_rf"]].round(4).to_string(index=False))

    # ---------- Q2 ----------
    print(f"\n===== Q2 4 default {int(df['pool_size'].max())} × configuration =====")
    q2 = (df.groupby(["metric", "selector"])
            .agg(n_targets=("target", "nunique"),
                 mean_regret_default_pool=("regret_default_pool", "mean"),
                 mean_regret_joint_pool=("regret_joint", "mean"),
                 max_regret_joint=("regret_joint", "max"),
                 n_nonzero_joint=("regret_joint", lambda s: int((s > 1e-9).sum())),
                 mean_ties=("n_ties_joint", "mean"),
                 max_ties=("n_ties_joint", "max"))
            .reset_index())
    q2["delta"] = q2["mean_regret_joint_pool"] - q2["mean_regret_default_pool"]
    print(q2.round(4).to_string(index=False))
    save_table(q2, outdir / "e04_q2_pool_expansion.csv")

    
    bad = df[df["n_ties_joint"] > 1]
    if len(bad):
        print(f"\n[] {len(bad)}/{len(df)} {bad['n_ties_joint'].max()}"
              f"\n metrics: {sorted(bad['metric'].unique())}"
              f"\n"
              f"\n 'CV X'")

    
    df["cv_picked_default_cfg"] = df["chosen_joint"].str.endswith("#00")
    print(f"\nCV 32 configurationdefault(#00):"
          f"{df['cv_picked_default_cfg'].mean():.2%}")

    write_run_meta(outdir, target="__cross_target__", stage="e04_summarize",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"targets": args.targets, "missing": missing,
                          "n_targets": int(n_t),
                          "notes": "F target target fold"
                                   "target 3–7 fold pparg"})
    print(f"\n[e04] write {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
