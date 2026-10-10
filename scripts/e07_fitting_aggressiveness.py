#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits                # noqa: E402

apply_thread_limits()

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy.stats import binomtest, spearmanr                     # noqa: E402

from pparbench.config import load_protocol                       # noqa: E402
from pparbench.io import RESULTS_DIR, save_table, write_run_meta # noqa: E402

import e01_regret_analysis as _e01                                # noqa: E402
import e08_bedroc_check as _e08                                  # noqa: E402



AGGRESSIVENESS = {"extratrees": 1, "rf": 2, "hgb": 3, "lightgbm": 4}
AXIS_MODELS = list(AGGRESSIVENESS)
PROTOCOL_ORDER = ["random_holdout", "scaffold_cv", "temporal"]



METRIC_SETS = {
    "e01": (_e01.METRICS, _e01.fold_mean_table, "e07_fitting_aggressiveness"),
    "e08": (_e08.METRICS, _e08.fold_mean_table, "e07_fitting_aggressiveness_m8"),
}
METRICS = _e01.METRICS          
_FOLD_MEAN = _e01.fold_mean_table
_CACHE: dict[str, pd.DataFrame] = {}


def _table(target: str) -> pd.DataFrame:
    """Documentation."""
    if target not in _CACHE:
        _CACHE[target] = _FOLD_MEAN(target).reset_index()
    return _CACHE[target]


def _rho(models: list[str], scores: list[float]) -> float:
    """Documentation."""
    if len(models) < 3:
        return float("nan")
    agg = [AGGRESSIVENESS[m] for m in models]
    r = spearmanr(agg, scores).statistic
    return float(-r) if np.isfinite(r) else float("nan")


def axis_table(targets: list[str]) -> pd.DataFrame:
    rows = []
    for t in targets:
        try:
            tab = _table(t)
        except FileNotFoundError:
            print(f"[warn] {t} no p03 predictions.csv skip")
            continue
        for protocol, g in tab.groupby("protocol"):
            s = g.set_index("model")
            present = [m for m in AXIS_MODELS if m in s.index]
            missing = [m for m in AXIS_MODELS if m not in s.index]
            if missing:
                print(f"[warn] {t}/{protocol} model {missing} ρ {len(present)} model")
            for metric in METRICS:
                scores = [float(s.loc[m, metric]) for m in present]
                rows.append(
                    {
                        "target": t,
                        "protocol": protocol,
                        "metric": metric,
                        "n_models": len(present),
                        "rho_less_aggressive_better": _rho(present, scores),
                        "best_model": present[int(np.argmax(scores))] if present else None,
                        "best_aggressiveness": (
                            AGGRESSIVENESS[present[int(np.argmax(scores))]] if present else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def sign_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for (protocol, metric), g in df.groupby(["protocol", "metric"]):
        r = g["rho_less_aggressive_better"].dropna()
        nz = r[r != 0]
        k = int((nz > 0).sum())
        n = int(len(nz))
        p = binomtest(k, n, 0.5).pvalue if n else float("nan")
        rows.append(
            {
                "protocol": protocol,
                "metric": metric,
                "n_targets": int(len(r)),
                "n_nonzero": n,
                "n_ties_rho_zero": int((r == 0).sum()),
                "k_less_aggressive_better": k,
                "frac": k / n if n else float("nan"),
                "mean_rho": float(r.mean()) if len(r) else float("nan"),
                "median_rho": float(r.median()) if len(r) else float("nan"),
                "p_two_sided": p,
            }
        )
    out = pd.DataFrame(rows)
    out["_po"] = out["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)})
    out["_mo"] = out["metric"].map({m: i for i, m in enumerate(METRICS)})
    return out.sort_values(["_mo", "_po"]).drop(columns=["_po", "_mo"]).reset_index(drop=True)


def monotonicity(st: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for metric, g in st.groupby("metric"):
        s = g.set_index("protocol")
        have = [p for p in PROTOCOL_ORDER if p in s.index]
        fr = [float(s.loc[p, "frac"]) for p in have]
        mr = [float(s.loc[p, "mean_rho"]) for p in have]
        rows.append(
            {
                "metric": metric,
                "protocols": "→".join(have),
                "frac_path": " → ".join(f"{v:.3f}" for v in fr),
                "mean_rho_path": " → ".join(f"{v:+.4f}" for v in mr),
                "frac_monotone_up": all(b >= a for a, b in zip(fr, fr[1:])),
                "mean_rho_monotone_up": all(b >= a for a, b in zip(mr, mr[1:])),
            }
        )
    return pd.DataFrame(rows)


def within_family(targets: list[str]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for t in targets:
        try:
            tab = _table(t)
        except FileNotFoundError:
            continue
        for protocol, g in tab.groupby("protocol"):
            s = g.set_index("model")
            if any(m not in s.index for m in AXIS_MODELS):
                continue
            for metric in METRICS:
                et, rf = float(s.loc["extratrees", metric]), float(s.loc["rf", metric])
                hgb, lgb = float(s.loc["hgb", metric]), float(s.loc["lightgbm", metric])
                rows.append(
                    {
                        "target": t,
                        "protocol": protocol,
                        "metric": metric,
                        "extratrees": et,
                        "rf": rf,
                        "hgb": hgb,
                        "lightgbm": lgb,
                        
                        "within_bagging_et_gt_rf": et > rf,
                        "within_boosting_hgb_gt_lgb": hgb > lgb,
                        "between_bag_gt_boost": (et + rf) / 2 > (hgb + lgb) / 2,
                    }
                )
    return pd.DataFrame(rows)


def within_family_summary(wf: pd.DataFrame) -> pd.DataFrame:
    cols = ["within_bagging_et_gt_rf", "within_boosting_hgb_gt_lgb", "between_bag_gt_boost"]
    rows = []
    for (protocol, metric), g in wf.groupby(["protocol", "metric"]):
        n = len(g)
        r = {"protocol": protocol, "metric": metric, "n_targets": n}
        for c in cols:
            k = int(g[c].sum())
            r[f"k_{c}"] = k
            r[f"p_{c}"] = binomtest(k, n, 0.5).pvalue if n else float("nan")
        rows.append(r)
    out = pd.DataFrame(rows)
    out["_po"] = out["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)})
    out["_mo"] = out["metric"].map({m: i for i, m in enumerate(METRICS)})
    return out.sort_values(["_mo", "_po"]).drop(columns=["_po", "_mo"]).reset_index(drop=True)


def main() -> None:
    global METRICS, _FOLD_MEAN

    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--metric-set", choices=sorted(METRIC_SETS), default="e01",
                    help="e01=4 metrics default e08=8 metrics BEDROC/ROC-AUC")
    args = ap.parse_args()

    METRICS, _FOLD_MEAN, outname = METRIC_SETS[args.metric_set]
    _CACHE.clear()
    print(f"[e07] metrics {args.metric_set} {len(METRICS)} {METRICS}")

    df = axis_table(args.targets)
    if df.empty:
        raise SystemExit("notargetavailable run first p03")
    st = sign_tests(df)
    mono = monotonicity(st)
    wf = within_family(args.targets)
    wfs = within_family_summary(wf)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    print("\n=== target 4 target ===")
    print(st.to_string(index=False))
    print("\n=== protocol ===")
    print(mono.to_string(index=False))
    print(f"\n=== vs k = {wfs['n_targets'].max()} target'' ===")
    print(wfs.to_string(index=False))

    print("\n--- ---")
    print("* n=4 model ρ {±1.0,±0.8,±0.4,±0.2,0}")
    print("* ridge/svr/knn")
    print("* results**** Guo 2024 supported''")
    print("'GEM ZINC15 training' training/GNN")
    print("message")
    a = st[(st["protocol"] == "temporal") & (st["metric"] == "spearman")]
    b = wfs[(wfs["protocol"] == "temporal") & (wfs["metric"] == "spearman")]
    if len(a) and len(b):
        ra, rb = a.iloc[0], b.iloc[0]
        print(
            f"\n* temporal/Spearman 4 {int(ra['k_less_aggressive_better'])}/"
            f"{int(ra['n_nonzero'])}，p={ra['p_two_sided']:.4f}"
        )
        print(
            f"bagging>boosting {int(rb['k_between_bag_gt_boost'])}/{int(rb['n_targets'])}"
            f" (p={rb['p_between_bag_gt_boost']:.4f})"
            f"| bagging et>rf {int(rb['k_within_bagging_et_gt_rf'])}/{int(rb['n_targets'])}"
            f"| boosting hgb>lgb {int(rb['k_within_boosting_hgb_gt_lgb'])}/{int(rb['n_targets'])}"
        )
        print("⇒ ** **")
        print("''")

    outdir = RESULTS_DIR / "cross_target" / outname
    outdir.mkdir(parents=True, exist_ok=True)
    save_table(df, outdir / "per_target_rho.csv")
    save_table(st, outdir / "sign_tests.csv")
    save_table(mono, outdir / "monotonicity.csv")
    save_table(wf, outdir / "within_family_per_target.csv")
    save_table(wfs, outdir / "within_family_summary.csv")
    write_run_meta(
        outdir,
        target="__cross_target__",
        stage="e07_fitting_aggressiveness",
        protocol_fingerprint=load_protocol().fingerprint,
        extra={
            "targets": args.targets,
            "metric_set": args.metric_set,
            "metrics": list(METRICS),
            "aggressiveness_order": AGGRESSIVENESS,
            "order_declared_from": "random// results",
            "axis_models_only": AXIS_MODELS,
            "excluded_models": ["ridge", "svr", "knn"],
            "purpose": "Guo & Ballester 2024 (arXiv:2406.00873)",
            "does_not_test": "GEM ZINC15 training",
            "ef_order_dependence": "enrich{5,10} argsort y_true"
                                   "(y_true, y_pred) note_8/10 §2"
                                   "e08 metrics bedroc20/rocauc kind='stable'"
                                   "message",
        },
    )
    print(f"\n[e07] write {outdir}")


if __name__ == "__main__":
    main()
