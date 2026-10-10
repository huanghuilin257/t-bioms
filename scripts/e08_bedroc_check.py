#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits  # noqa: E402

apply_thread_limits()  

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import target_results_dir, save_table, write_run_meta  # noqa: E402
from pparbench.metrics import enrichment_factor  # noqa: E402

from e01_regret_analysis import POOL_TREES, SELECTORS, TEMPORAL, regret  # noqa: E402

ALPHA = 20.0
PCTS = (5, 10)

METRICS = ["spearman", "kendall",
           "enrich5", "bedroc20_top5", "rocauc_top5",
           "enrich10", "bedroc20_top10", "rocauc_top10"]


# ----------------------------------------------------------------- BEDROC
def bedroc(labels_by_rank: np.ndarray, alpha: float = ALPHA) -> float:
    """Documentation."""
    lab = np.asarray(labels_by_rank, dtype=float)
    N = len(lab)
    n = int(lab.sum())
    if n == 0 or n == N:
        return float("nan")
    ra = n / N
    ranks = np.flatnonzero(lab > 0) + 1  # 1-indexed

    rie_num = np.exp(-alpha * ranks / N).sum() / n
    rie_den = (1.0 / N) * (1.0 - np.exp(-alpha)) / (np.exp(alpha / N) - 1.0)
    rie = rie_num / rie_den

    scale = (ra * np.sinh(alpha / 2.0)
             / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra)))
    return float(rie * scale + 1.0 / (1.0 - np.exp(alpha * (1.0 - ra))))


def _top_k_labels(y_true: np.ndarray, pct: float) -> tuple[np.ndarray, int, int]:
    """Documentation."""
    n = len(y_true)
    k = max(1, int(round(n * pct / 100.0)))
    order = np.argsort(-y_true, kind="stable")
    lab = np.zeros(n, dtype=int)
    lab[order[:k]] = 1
    cutoff = y_true[order[k - 1]]
    n_tie_boundary = int((y_true == cutoff).sum())
    return lab, k, n_tie_boundary


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Documentation."""
    if len(y_true) < 3 or np.all(y_pred == y_pred[0]):
        return {m: np.nan for m in METRICS} | {f"n_tie_boundary{p}": np.nan for p in PCTS}
    out: dict[str, float] = {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
    }
    pred_order = np.argsort(-y_pred, kind="stable")
    for pct in PCTS:
        lab, _k, n_tie = _top_k_labels(y_true, pct)
        out[f"enrich{pct}"] = enrichment_factor(y_true, y_pred, pct)
        out[f"bedroc20_top{pct}"] = bedroc(lab[pred_order], ALPHA)
        out[f"rocauc_top{pct}"] = float(roc_auc_score(lab, y_pred))
        out[f"n_tie_boundary{pct}"] = float(n_tie)
    return out


def fold_mean_table(slug: str) -> pd.DataFrame:
    """Documentation."""
    p = target_results_dir(slug, "p03_protocol_matrix", create=False) / "predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    pred = pd.read_csv(p)
    rows = []
    for (protocol, model, fold), g in pred.groupby(["protocol", "model", "fold"]):
        rows.append({"protocol": protocol, "model": model, "fold": fold, "n": len(g),
                     **_metrics(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))})
    per_fold = pd.DataFrame(rows)
    cols = METRICS + [f"n_tie_boundary{p}" for p in PCTS]
    agg = per_fold.groupby(["protocol", "model"])[cols].mean().reset_index()
    agg["n_folds"] = per_fold.groupby(["protocol", "model"])["fold"].nunique().to_numpy()
    agg["target"] = slug
    return agg



def _selftest() -> None:
    """Documentation."""
    from rdkit.ML.Scoring import Scoring

    rng = np.random.RandomState(66)
    worst = 1e9
    for trial in range(5):
        N, n = 400, 20
        lab = np.zeros(N, dtype=int)
        lab[rng.choice(N, n, replace=False)] = 1
        score = rng.rand(N)
        order = np.argsort(-score, kind="stable")
        by_rank = lab[order]
        mine = bedroc(by_rank, ALPHA)
        ref = Scoring.CalcBEDROC([[int(v)] for v in by_rank], 0, ALPHA)
        worst = min(worst, abs(mine - ref))
        assert abs(mine - ref) < 1e-9, f"trial {trial}: {mine} vs rdkit {ref}"
    perfect = bedroc(np.r_[np.ones(20), np.zeros(380)].astype(int), ALPHA)
    reverse = bedroc(np.r_[np.zeros(380), np.ones(20)].astype(int), ALPHA)
    assert perfect > 0.999, perfect
    assert reverse < 1e-6, reverse
    print(f"[e08] OK BEDROC rdkit consistent 5/5 <1e-9"
          f"{perfect:.6f} {reverse:.2e}")


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="e08 BEDROC/ROC-AUC enrichmentmetrics")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    _selftest()

    prot = load_protocol()
    full_pool = list(prot.models)

    if args.targets:
        slugs = args.targets
    else:
        rd = ROOT / "results"
        slugs = sorted(d.name for d in rd.iterdir()
                       if d.is_dir() and d.name != "cross_target"
                       and (d / "p03_protocol_matrix" / "predictions.csv").exists())
    print(f"[e08] target {len(slugs)}: {slugs}")
    print(f"[e08] {full_pool}\n[e08] {POOL_TREES}")

    aggs = []
    for slug in slugs:
        agg = fold_mean_table(slug)
        ref = pd.read_csv(target_results_dir(slug, "p03_protocol_matrix",
                                             create=False) / "aggregated.csv")
        m = agg.merge(ref[["protocol", "model", "spearman_foldmean"]],
                      on=["protocol", "model"], how="inner")
        d = (m["spearman"] - m["spearman_foldmean"]).abs().max()
        assert d < 5e-4, f"{slug} spearman failed {d:.6f}"
        print(f"[e08] {slug:9s} {len(agg):3d} OK")
        aggs.append(agg)

    all_agg = pd.concat(aggs, ignore_index=True)
    out = ROOT / "results" / "cross_target" / "e08_bedroc"
    out.mkdir(parents=True, exist_ok=True)
    save_table(all_agg, out / "e08_foldmean_all_metrics.csv")

    rows = []
    for slug, agg in all_agg.groupby("target"):
        for metric in METRICS:
            for selector in SELECTORS:
                for pname, pool in (("trees4", POOL_TREES), ("all", full_pool)):
                    r = regret(agg, selector=selector, pool=pool, metric=metric)
                    if r:
                        rows.append({"target": slug, "pool": pname, **r})
    reg = pd.DataFrame(rows)
    save_table(reg, out / "e08_regret.csv")

    
    NONTREE = [m for m in full_pool if m not in POOL_TREES]
    print(f"\n================ pool=all CV model {NONTREE} target ================")
    a = reg[reg["pool"] == "all"]
    tab = []
    for metric in METRICS:
        for selector in SELECTORS:
            s = a[(a["metric"] == metric) & (a["selector"] == selector)]
            if not len(s):
                continue
            tab.append({
                "metric": metric, "selector": selector, "n_targets": len(s),
                "cv_picks_nontree": int(s["chosen"].isin(NONTREE).sum()),
                "truth_is_nontree": int(s["temporal_best_model"].isin(NONTREE).sum()),
                "picked": "/".join(sorted(set(s.loc[s["chosen"].isin(NONTREE), "chosen"]))) or "-",
            })
    print(pd.DataFrame(tab).to_string(index=False))

    print("\n================ trees4 ================")
    d_t = (reg[reg["pool"] == "trees4"]
           .groupby(["metric", "selector"])[["regret", "random_pick_regret"]].mean())
    d_a = (reg[reg["pool"] == "all"]
           .groupby(["metric", "selector"])[["regret", "random_pick_regret"]].mean())
    cmp = pd.DataFrame({
        "num_trees4": d_t["regret"], "num_all": d_a["regret"],
        "den_trees4": d_t["random_pick_regret"], "den_all": d_a["random_pick_regret"],
    })
    cmp["ratio_trees4"] = cmp["num_trees4"] / cmp["den_trees4"]
    cmp["ratio_all_native"] = cmp["num_all"] / cmp["den_all"]
    cmp["ratio_all_fixed_den"] = cmp["num_all"] / cmp["den_trees4"]
    print(cmp.round(4).to_string())
    print("⚠ molecules 11")

    print("\n================ EF/BEDROC ================")
    t = all_agg[all_agg["protocol"] == TEMPORAL]
    for pct in PCTS:
        c = f"n_tie_boundary{pct}"
        print(f"top{pct}%: mean {t[c].mean():.2f} {t[c].max():.1f}"
              f"=1")
    print(t.groupby("target")[[f"n_tie_boundary{p}" for p in PCTS]].mean()
          .round(2).to_string())

    print("\n================ target pool=all / random_holdout ================")
    s = a[a["selector"] == "random_holdout"]
    print(s.pivot_table(index="target", columns="metric", values="chosen",
                        aggfunc="first").reindex(columns=METRICS).to_string())

    write_run_meta(out, target="__cross_target__", stage="e08_bedroc_check",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"targets": slugs, "full_pool": full_pool,
                          "pool_trees": POOL_TREES, "metrics": METRICS,
                          "alpha": ALPHA, "actives": "y_true top-k, k=5%/10%",
                          "bedroc_validated_against": "rdkit.ML.Scoring.CalcBEDROC",
                          "notes": "note_8/09 §6 2 enrichmentmetrics EF"
                                   "input p03 predictions.csv"
                                   "regret"})
    print(f"\n[e08] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
