#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()  

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from pparbench.config import load_protocol             # noqa: E402
from pparbench.io import save_table, write_run_meta     # noqa: E402
from e01_regret_analysis import fold_mean_table         # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
FROZEN20_CSV = (ROOT / "results" / "cross_target" / "e01_regret"
                / "e01_foldmean_all_metrics.csv")
OUT_DIR = ROOT / "results" / "cross_target" / "e15_landrum_preflight"

TEMPORAL = "temporal"
SELECTOR = "random_holdout"          
POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]
METRICS = ["spearman", "kendall", "enrich10", "enrich5"]
HEADLINE = ("trees4", "spearman")




NEW_FULL7 = ["htr1a", "adra1d", "drd3", "slc6a3", "htr6", "hrh4"]
NEW_PARTIAL = {"mtnr1a": 4, "trpv1": 2}
NEW_DROPPED = {"hcv_ns3": 357, "trpv1_rat": 428, "dpp4": 549, "hiv1_pr": 699}
NEW_USABLE = NEW_FULL7 + list(NEW_PARTIAL)



CRIT_GO = 47
CRIT_GO_ORIGINAL = 73                
N_ELIGIBLE_NEW = 53                  
N_FROZEN = 20


def _pick(s: pd.Series, pool: list[str]) -> tuple[str, int]:
    """Documentation."""
    s = s.reindex(pool).dropna()
    if s.empty:
        return "", 0
    best = s.max()
    ties = [m for m in pool if m in s.index and np.isclose(s[m], best, atol=1e-12)]
    return ties[0], len(ties)


def prescribe_from_frozen20(frozen: pd.DataFrame, *, pool: list[str],
                            metric: str) -> tuple[str, int, pd.Series]:
    """Documentation."""
    train = frozen[frozen["protocol"] == SELECTOR]
    norm = []
    for _t, g in train.groupby("target"):
        s = g.set_index("model")[metric].reindex(pool).dropna()
        if len(s) < 2 or not np.isfinite(s.max() - s.min()) or s.max() == s.min():
            continue
        norm.append((s - s.min()) / (s.max() - s.min()))
    if not norm:
        return "", 0, pd.Series(dtype=float)
    mean_norm = pd.concat(norm, axis=1).mean(axis=1)
    m_star, n_ties = _pick(mean_norm, pool)
    return m_star, n_ties, mean_norm


def per_target_delta(new_agg: pd.DataFrame, *, target: str, pool: list[str],
                     metric: str, m_star: str) -> dict | None:
    """Documentation."""
    piv = new_agg[new_agg["target"] == target].pivot_table(
        index="model", columns="protocol", values=metric)
    if SELECTOR not in piv.columns or TEMPORAL not in piv.columns:
        return None
    tmp = piv[TEMPORAL].reindex(pool).dropna()
    sel = piv[SELECTOR].reindex(pool).dropna()
    if tmp.empty or sel.empty or m_star not in tmp.index:
        return None

    tbest, _ = _pick(tmp, pool)
    selfpick, n_ties_self = _pick(sel, pool)
    if selfpick not in tmp.index:
        return None

    r_star = float(tmp[tbest] - tmp[m_star])
    r_self = float(tmp[tbest] - tmp[selfpick])
    avail = [m for m in pool if m in tmp.index]
    return {
        "target": target, "metric": metric,
        "n_cutpoints": 7 if target in NEW_FULL7 else NEW_PARTIAL.get(target, np.nan),
        "full7": target in NEW_FULL7,
        "prescribed": m_star, "prescribed_temporal": float(tmp[m_star]),
        "cv_selfpick": selfpick, "selfpick_temporal": float(tmp[selfpick]),
        "n_ties_selfpick": n_ties_self,
        "temporal_best_model": tbest, "temporal_best": float(tmp[tbest]),
        "regret_prescribed": r_star, "regret_cv_selfpick": r_self,
        "delta": r_self - r_star,
        "random_pick_regret": float(np.mean([tmp[tbest] - tmp[m] for m in avail])),
        "same_model": selfpick == m_star,
    }


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


def main() -> int:
    prot = load_protocol()
    pools = {"trees4": POOL_TREES, "all7": list(prot.models)}

    
    if not FROZEN20_CSV.exists():
        print(f"[e15] {FROZEN20_CSV}", file=sys.stderr)
        return 1
    frozen = pd.read_csv(FROZEN20_CSV)
    n_ref = frozen["target"].nunique()
    print(f"[e15] = {n_ref} target e01_foldmean_all_metrics.csv")
    if n_ref != N_FROZEN:
        print(f"[e15] !! {N_FROZEN} target {n_ref}"
              f"e01 target 35", file=sys.stderr)
        return 1
    if set(frozen["target"]) & set(NEW_USABLE):
        print("[e15] !! target", file=sys.stderr)
        return 1

    
    from pparbench.io import target_results_dir
    aggs = []
    for slug in NEW_USABLE:
        agg = fold_mean_table(slug)
        ref = pd.read_csv(target_results_dir(slug, "p03_protocol_matrix",
                                            create=False) / "aggregated.csv")
        m = agg.merge(ref[["protocol", "model", "spearman_foldmean"]],
                      on=["protocol", "model"], how="inner")
        dmax = (m["spearman"] - m["spearman_foldmean"]).abs().max()
        flag = "OK" if dmax < 5e-4 else f"!! {dmax:.5f}"
        nf = agg[agg["protocol"] == TEMPORAL]["n_folds"].max()
        print(f"[e15] {slug:9s} {len(agg):3d} temporal fold ={nf} {flag}")
        if dmax >= 5e-4:
            return 1
        aggs.append(agg)
    new_agg = pd.concat(aggs, ignore_index=True)
    save_table(new_agg, OUT_DIR / "e15_new_foldmean_all_metrics.csv", float_fmt="%.6g")

    
    presc_rows, delta_rows = [], []
    for pool_name, pool in pools.items():
        for metric in METRICS:
            m_star, n_ties, mean_norm = prescribe_from_frozen20(
                frozen, pool=pool, metric=metric)
            presc_rows.append({
                "pool": pool_name, "metric": metric, "source": SELECTOR,
                "prescribed": m_star, "n_ties": n_ties,
                **{f"meannorm_{m}": float(mean_norm.get(m, np.nan)) for m in pool},
            })
            for slug in NEW_USABLE:
                r = per_target_delta(new_agg, target=slug, pool=pool,
                                     metric=metric, m_star=m_star)
                if r:
                    delta_rows.append({"pool": pool_name, **r})
    presc = pd.DataFrame(presc_rows)
    dl = pd.DataFrame(delta_rows)
    save_table(presc, OUT_DIR / "e15_prescription.csv", float_fmt="%.6g")
    save_table(dl, OUT_DIR / "e15_delta_by_target.csv", float_fmt="%.6g")

    
    rows = []
    for (pool_name, metric), g in dl.groupby(["pool", "metric"]):
        for label, sub in (("full7_n6", g[g["full7"]]), ("all_usable_n8", g)):
            rows.append({"pool": pool_name, "metric": metric,
                         **power_row(sub["delta"].to_numpy(), label),
                         "targets": ";".join(sorted(sub["target"]))})
    power = pd.DataFrame(rows)
    save_table(power, OUT_DIR / "e15_power.csv", float_fmt="%.6g")

    
    hp, hm = HEADLINE
    verdicts = []
    for label in ("full7_n6", "all_usable_n8"):
        r = power[(power["pool"] == hp) & (power["metric"] == hm)
                  & (power["subset"] == label)]
        tn = float(r["t_needed"].iloc[0]) if len(r) else np.nan
        if not np.isfinite(tn):
            v = "Δ̄≈0 no target"
        elif tn <= CRIT_GO:
            v = (f"1 T_needed={tn:.0f} ≲ {CRIT_GO} H1"
                 f"~{CRIT_GO}")
        else:
            v = (f"3 T_needed={tn:.0f} > {CRIT_GO} ****"
                 f"results")
        verdicts.append({"subset": label, "pool": hp, "metric": hm,
                         "t_needed": tn, "verdict": v})
    vd = pd.DataFrame(verdicts)

    
    surv_full7 = len(NEW_FULL7) / (len(NEW_USABLE) + len(NEW_DROPPED))
    surv_any = len(NEW_USABLE) / (len(NEW_USABLE) + len(NEW_DROPPED))
    reach = pd.DataFrame([
        {"basis": "7", "survival": surv_full7,
         "projected_new": int(round(N_ELIGIBLE_NEW * surv_full7)),
         "projected_total": N_FROZEN + int(round(N_ELIGIBLE_NEW * surv_full7))},
        {"basis": "message", "survival": surv_any,
         "projected_new": int(round(N_ELIGIBLE_NEW * surv_any)),
         "projected_total": N_FROZEN + int(round(N_ELIGIBLE_NEW * surv_any))},
    ])
    save_table(vd, OUT_DIR / "e15_verdict.csv", float_fmt="%.6g")
    save_table(reach, OUT_DIR / "e15_reachability.csv", float_fmt="%.6g")

    
    print(f"\n[e15] {hp} × {hm} ={SELECTOR} 20 target :"
          f"{presc[(presc['pool'] == hp) & (presc['metric'] == hm)]['prescribed'].iloc[0]}")
    d = dl[(dl["pool"] == hp) & (dl["metric"] == hm)]
    print(f"\n[e15] {hp} × {hm} :")
    print(d[["target", "n_cutpoints", "prescribed", "cv_selfpick", "same_model",
             "regret_prescribed", "regret_cv_selfpick", "delta"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[e15] {hp} × {hm} :")
    print(power[(power["pool"] == hp) & (power["metric"] == hm)]
          [["subset", "n_targets", "n_zero_delta", "n_positive", "n_negative",
            "delta_mean", "delta_sd", "t_needed", "sd_rel_se"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[e15] §3.1 :")
    for _, r in vd.iterrows():
        print(f"  {r['subset']:14s} {r['verdict']}")
    print("\n[e15] 40 T_needed failed :")
    print(reach.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[e15] molecules < 750 39 :"
          f"{', '.join(f'{k}({v})' for k, v in NEW_DROPPED.items())}")

    write_run_meta(OUT_DIR, target="cross_target", stage="e15_landrum_preflight",
                   protocol_fingerprint=prot.fingerprint,
                   extra={
                       "declared_in": DECLARED_IN,
                       "reference_pool": "frozen20 (read-only from e01_regret)",
                       "n_reference_targets": int(n_ref),
                       "new_full7": NEW_FULL7,
                       "new_partial": NEW_PARTIAL,
                       "new_dropped_no_valid_cutpoint": NEW_DROPPED,
                       "selector_source": SELECTOR,
                       "headline": {"pool": hp, "metric": hm, "subset": "full7_n6"},
                       "criteria_frozen": {"go": CRIT_GO,
                                           "go_original_before_amendment": CRIT_GO_ORIGINAL,
                                           "amendment_reason": "§7.4 42"},
                       "n_eligible_new_landrum": N_ELIGIBLE_NEW,
                       "note": "RNG",
                   })
    print(f"\n[e15] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
