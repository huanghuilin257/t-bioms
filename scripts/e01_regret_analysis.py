#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.config import load_protocol           # noqa: E402
from pparbench.io import target_results_dir, save_table, write_run_meta  # noqa: E402
from pparbench.metrics import enrichment_factor      # noqa: E402

from scipy.stats import kendalltau, spearmanr        # noqa: E402

SELECTORS = ["random_holdout", "scaffold_cv"]
TEMPORAL = "temporal"

POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]






FROZEN20 = [
    "abl1", "ache", "adora2a", "ar", "bace1", "drd2", "egfr", "esr1", "f10", "f2",
    "hdac1", "htr2a", "kcnh2", "kdr", "nr3c1", "oprm1", "pik3ca", "ppara", "ppard", "pparg",
]



def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Documentation."""
    if len(y_true) < 3 or np.all(y_pred == y_pred[0]):
        return {"spearman": np.nan, "kendall": np.nan,
                "enrich5": np.nan, "enrich10": np.nan}
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
        "enrich5": enrichment_factor(y_true, y_pred, 5),
        "enrich10": enrichment_factor(y_true, y_pred, 10),
    }


METRICS = ["spearman", "kendall", "enrich5", "enrich10"]


def fold_mean_table(slug: str) -> pd.DataFrame:
    """Documentation."""
    p = target_results_dir(slug, "p03_protocol_matrix", create=False) / "predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    pred = pd.read_csv(p)

    rows = []
    for (protocol, model, fold), g in pred.groupby(["protocol", "model", "fold"]):
        rows.append({"protocol": protocol, "model": model, "fold": fold,
                     "n": len(g), **_metrics(g["y_true"].to_numpy(),
                                             g["y_pred"].to_numpy())})
    per_fold = pd.DataFrame(rows)

    agg = (per_fold.groupby(["protocol", "model"])[METRICS]
           .mean().reset_index())
    agg["n_folds"] = (per_fold.groupby(["protocol", "model"])["fold"]
                      .nunique().to_numpy())
    agg["target"] = slug
    return agg



def _pick(scores: dict[str, float], pool: list[str]) -> tuple[str, int]:
    """Documentation."""
    avail = [m for m in pool if m in scores and np.isfinite(scores[m])]
    if not avail:
        return "", 0
    best = max(scores[m] for m in avail)
    ties = [m for m in avail if np.isclose(scores[m], best, atol=1e-12)]
    return ties[0], len(ties)


def regret(agg: pd.DataFrame, *, selector: str, pool: list[str],
           metric: str) -> dict[str, float | str | int]:
    """Documentation."""
    piv = agg.pivot_table(index="model", columns="protocol", values=metric)
    if selector not in piv.columns or TEMPORAL not in piv.columns:
        return {}
    sel_scores = piv[selector].dropna().to_dict()
    tmp_scores = piv[TEMPORAL].dropna().to_dict()

    chosen, n_ties = _pick(sel_scores, pool)
    tbest, _ = _pick(tmp_scores, pool)
    if not chosen or not tbest or chosen not in tmp_scores:
        return {}

    avail = [m for m in pool if m in tmp_scores]
    
    
    random_pick = float(np.mean([tmp_scores[tbest] - tmp_scores[m] for m in avail]))

    return {
        "selector": selector, "metric": metric, "pool_size": len(avail),
        "chosen": chosen, "chosen_selector_score": float(sel_scores[chosen]),
        "chosen_temporal": float(tmp_scores[chosen]),
        "temporal_best_model": tbest,
        "temporal_best": float(tmp_scores[tbest]),
        "regret": float(tmp_scores[tbest] - tmp_scores[chosen]),
        "random_pick_regret": random_pick,
        "n_ties": n_ties,
    }



def loto(all_agg: pd.DataFrame, *, pool: list[str], metric: str) -> pd.DataFrame:
    """Documentation."""
    tmp = all_agg[all_agg["protocol"] == TEMPORAL]
    targets = sorted(tmp["target"].unique())
    rows = []
    for held in targets:
        train = tmp[tmp["target"] != held]
        
        
        norm = []
        for t, g in train.groupby("target"):
            s = g.set_index("model")[metric].reindex(pool).dropna()
            if len(s) < 2 or not np.isfinite(s.max() - s.min()) or s.max() == s.min():
                continue
            norm.append((s - s.min()) / (s.max() - s.min()))
        if not norm:
            continue
        mean_rank = pd.concat(norm, axis=1).mean(axis=1)
        m_star = str(mean_rank.idxmax())

        g = tmp[tmp["target"] == held].set_index("model")[metric]
        avail = [m for m in pool if m in g.index and np.isfinite(g[m])]
        if m_star not in avail:
            continue
        tbest = max(avail, key=lambda m: g[m])
        rows.append({
            "held_out": held, "metric": metric, "prescribed": m_star,
            "prescribed_temporal": float(g[m_star]),
            "temporal_best_model": tbest, "temporal_best": float(g[tbest]),
            "regret": float(g[tbest] - g[m_star]),
        })
    return pd.DataFrame(rows)



def poolsize_curve(all_agg: pd.DataFrame, *, full_pool: list[str],
                   metric: str, selector: str) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for slug, agg in all_agg.groupby("target"):
        present = [m for m in full_pool
                   if m in set(agg[agg["protocol"] == TEMPORAL]["model"])]
        for k in range(2, len(present) + 1):
            vals = []
            for sub in itertools.combinations(present, k):
                r = regret(agg, selector=selector, pool=list(sub), metric=metric)
                if r:
                    vals.append(r["regret"])
            if vals:
                rows.append({"target": slug, "metric": metric, "selector": selector,
                             "pool_size": k, "n_subsets": len(vals),
                             "regret_mean": float(np.mean(vals)),
                             "regret_max": float(np.max(vals)),
                             "frac_nonzero": float(np.mean(np.asarray(vals) > 1e-9))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="e01 C/D/E")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    prot = load_protocol()
    full_pool = list(prot.models)          

    if args.targets:
        slugs = args.targets
    else:
        rd = ROOT / "results"
        slugs = sorted(d.name for d in rd.iterdir()
                       if d.is_dir() and d.name != "cross_target"
                       and (d / "p03_protocol_matrix" / "predictions.csv").exists())
        
        
        if set(slugs) != set(FROZEN20):
            extra = sorted(set(slugs) - set(FROZEN20))
            missing = sorted(set(FROZEN20) - set(slugs))
            print(f"[e01] !! targetinconsistent")
            print(f"[e01] {len(slugs)} {len(FROZEN20)}")
            if extra:
                print(f"[e01] target : {extra}")
            if missing:
                print(f"[e01] : {missing}")
            print(f"[e01] note_8/19 §2 20 target")
            print(f"[e01] : --targets results/ 20")
            print(f"[e01] : --targets {' '.join(FROZEN20)}")
            print(f"[e01] target: --targets output")
            return 1
    print(f"[e01] target {len(slugs)}: {slugs}")
    print(f"[e01] {full_pool}")
    print(f"[e01] {POOL_TREES}")

    aggs = []
    for slug in slugs:
        agg = fold_mean_table(slug)
        
        ref = pd.read_csv(target_results_dir(slug, "p03_protocol_matrix",
                                             create=False) / "aggregated.csv")
        m = agg.merge(ref[["protocol", "model", "spearman_foldmean"]],
                      on=["protocol", "model"], how="inner")
        d = (m["spearman"] - m["spearman_foldmean"]).abs().max()
        flag = "OK" if d < 5e-4 else f"!! {d:.5f}"
        print(f"[e01] {slug:9s} {len(agg):3d} {flag}")
        aggs.append(agg)

    all_agg = pd.concat(aggs, ignore_index=True)
    
    
    #
    
    
    
    
    if set(slugs) == set(FROZEN20):
        out = ROOT / "results" / "cross_target" / "e01_regret"
    else:
        out = ROOT / "results" / "cross_target" / f"e01_regret__offpool_n{len(slugs)}"
        print(f"[e01] !! target 20 output {out.name}/")
        print(f"[e01] e01_regret/")
    out.mkdir(parents=True, exist_ok=True)
    save_table(all_agg, out / "e01_foldmean_all_metrics.csv")

    
    rows = []
    for slug, agg in all_agg.groupby("target"):
        for metric in METRICS:
            for selector in SELECTORS:
                for pname, pool in (("trees4", POOL_TREES), ("all", full_pool)):
                    r = regret(agg, selector=selector, pool=pool, metric=metric)
                    if r:
                        rows.append({"target": slug, "pool": pname, **r})
    reg = pd.DataFrame(rows)
    save_table(reg, out / "e01_regret.csv")

    # ---- C：LOTO ----
    lo = pd.concat([loto(all_agg, pool=POOL_TREES, metric=m) for m in METRICS],
                   ignore_index=True)
    save_table(lo, out / "e01_loto.csv")

    
    ps = poolsize_curve(all_agg, full_pool=full_pool,
                        metric="spearman", selector="random_holdout")
    save_table(ps, out / "e01_poolsize.csv")

    
    print("\n================ D metrics target ================")
    for pname in ("trees4", "all"):
        print(f"\n--- = {pname} ---")
        piv = (reg[reg["pool"] == pname]
               .pivot_table(index="metric", columns="selector",
                            values=["regret", "random_pick_regret"],
                            aggfunc="mean"))
        print(piv.round(4).to_string())
        for selector in SELECTORS:
            s = reg[(reg["pool"] == pname) & (reg["selector"] == selector)]
            for metric in METRICS:
                v = s[s["metric"] == metric]["regret"]
                if len(v):
                    print(f"    {selector:16s} {metric:9s} "
                          f"mean {v.mean():.4f} {v.max():.4f}"
                          f"{(v > 1e-9).sum()}/{len(v)}")

    print("\n================ C LOTO model ================")
    for metric in METRICS:
        s = lo[lo["metric"] == metric]
        if not len(s):
            continue
        pres = s["prescribed"].value_counts().to_dict()
        print(f"{metric:9s} mean {s['regret'].mean():.4f}"
              f"{s['regret'].max():.4f} {(s['regret'] > 1e-9).sum()}/{len(s)}"
              f"{pres}")
    print("\n target spearman")
    print(lo[lo["metric"] == "spearman"][
        ["held_out", "prescribed", "prescribed_temporal",
         "temporal_best_model", "temporal_best", "regret"]].to_string(index=False))

    print("\n================ E ================")
    print(ps.groupby("pool_size")[["regret_mean", "regret_max", "frac_nonzero"]]
          .mean().round(4).to_string())

    print("\n================ B ================")
    t = reg[(reg["pool"] == "trees4") & (reg["metric"] == "spearman")]
    print(t[["target", "selector", "chosen", "n_ties"]].to_string(index=False))

    write_run_meta(out, target="__cross_target__", stage="e01_regret_analysis",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"targets": slugs, "full_pool": full_pool,
                          "pool_trees": POOL_TREES, "metrics": METRICS,
                          "selectors": SELECTORS,
                          "notes": "note_8/04 C/D/E"
                                   "input p03 predictions.csv"})
    print(f"\n[e01] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
