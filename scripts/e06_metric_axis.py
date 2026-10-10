#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from scipy.stats import binomtest                               # noqa: E402

from pparbench.config import load_protocol                       # noqa: E402
from pparbench.io import RESULTS_DIR, save_table, write_run_meta # noqa: E402
from pparbench.models import MODEL_FAMILY                        # noqa: E402

from e01_regret_analysis import METRICS, fold_mean_table         # noqa: E402

RANKING = ["spearman", "kendall"]
EARLY = ["enrich5", "enrich10"]
BAGGING = [m for m, f in MODEL_FAMILY.items() if f == "bagging"]
BOOSTING = [m for m, f in MODEL_FAMILY.items() if f == "boosting"]


def family_deltas(targets: list[str]) -> pd.DataFrame:
    rows = []
    for t in targets:
        try:
            agg = fold_mean_table(t).reset_index()
        except FileNotFoundError:
            print(f"[warn] {t} no p03 predictions.csv skip")
            continue
        for protocol, g in agg.groupby("protocol"):
            s = g.set_index("model")
            for metric in METRICS:
                bag = [s.loc[m, metric] for m in BAGGING if m in s.index]
                boo = [s.loc[m, metric] for m in BOOSTING if m in s.index]
                bag = [v for v in bag if np.isfinite(v)]
                boo = [v for v in boo if np.isfinite(v)]
                if not bag or not boo:
                    continue
                rows.append({"target": t, "protocol": protocol, "metric": metric,
                             "kind": "ranking" if metric in RANKING else "early",
                             "bagging": float(np.mean(bag)),
                             "boosting": float(np.mean(boo)),
                             "delta": float(np.mean(bag) - np.mean(boo))})
    return pd.DataFrame(rows)


def sign_test(d: pd.DataFrame, label: str) -> dict:
    """Documentation."""
    nz = d[d["delta"].abs() > 1e-12]
    n, k = len(nz), int((nz["delta"] > 0).sum())
    p = binomtest(k, n, 0.5).pvalue if n else np.nan
    print(f"{label:34s} bagging {k:2d}/{n:2d} p={p:.4f}"
          f"={d['delta'].mean():+.4f}")
    return {"label": label, "n": n, "n_bagging_better": k, "p_value": p,
            "mean_delta": float(d["delta"].mean()),
            "n_zero_excluded": int(len(d) - n)}


def main() -> int:
    ap = argparse.ArgumentParser(description="e06 metrics")
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    df = family_deltas(args.targets)
    if df.empty:
        print("[error] noavailabledata")
        return 1
    n_t = df["target"].nunique()

    outdir = RESULTS_DIR / "cross_target" / "e06_metric_axis"
    outdir.mkdir(parents=True, exist_ok=True)
    save_table(df, outdir / "family_deltas.csv")

    print(f"\n===== = bagging − boosting {n_t} target target =====")
    print("= bagging (protocol, metrics) group \n")
    summary = []
    for protocol in ["random_holdout", "scaffold_cv", "temporal"]:
        sub = df[df["protocol"] == protocol]
        if sub.empty:
            continue
        print(f"[{protocol}]")
        for metric in METRICS:
            m = sub[sub["metric"] == metric]
            if len(m):
                r = sign_test(m, metric)
                r.update(protocol=protocol, metric=metric, kind=m["kind"].iloc[0])
                summary.append(r)
        print()

    print("===== temporal metrics vs metrics =====")
    print("⚠ 2 metrics × 9 target n=18"
          "target Spearman Kendall ** ** \n"
          "p metrics 8/9 p=0.039")
    tmp = df[df["protocol"] == "temporal"]
    for kind in ("ranking", "early"):
        r = sign_test(tmp[tmp["kind"] == kind], f"temporal / {kind}")
        r.update(protocol="temporal", metric=f"__{kind}__", kind=kind)
        summary.append(r)

    
    piv = (tmp.pivot_table(index="target", columns="kind", values="delta")
           .reset_index())
    if {"ranking", "early"} <= set(piv.columns):
        piv["reversed"] = np.sign(piv["ranking"]) != np.sign(piv["early"])
        print("\ntarget temporal delta = bagging − boosting")
        print(piv.round(4).to_string(index=False))
        n_rev = int(piv["reversed"].sum())
        n_bag_rank = int((piv["ranking"] > 0).sum())
        n_bag_early = int((piv["early"] > 0).sum())
        print(f"\n⇒ {n_rev}/{len(piv)} targetmetrics****")
        print(f"⇒ metrics bagging target {n_bag_rank}/{len(piv)}"
              f"metrics bagging target {n_bag_early}/{len(piv)}")
        if n_bag_rank > n_bag_early:
            print("⇒ metrics**consistent** metrics bagging")
        elif n_bag_rank == n_bag_early:
            print("⇒ metrics ⇒ metricsdata**nosupported**")
        else:
            print("⇒ **** ⇒ data note_8/05 §4.3")
        save_table(piv, outdir / "per_target_reversal.csv")

    save_table(pd.DataFrame(summary), outdir / "sign_tests.csv")
    write_run_meta(outdir, target="__cross_target__", stage="e06_metric_axis",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"targets": args.targets, "n_targets": int(n_t),
                          "ranking_metrics": RANKING, "early_metrics": EARLY,
                          "bagging": BAGGING, "boosting": BOOSTING,
                          "notes": "note_8/05 §4.3 metrics target"
                                   "Spearman enrichment"})
    print(f"\n[e06] write {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
