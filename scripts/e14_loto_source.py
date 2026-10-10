#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()  

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from scipy.stats import wilcoxon                       # noqa: E402

from pparbench.config import load_protocol             # noqa: E402
from pparbench.io import save_table, write_run_meta     # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
IN_CSV = ROOT / "results" / "cross_target" / "e01_regret" / "e01_foldmean_all_metrics.csv"
OUT_DIR = ROOT / "results" / "cross_target" / "e14_loto_source"

TEMPORAL = "temporal"
POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]
METRICS = ["spearman", "kendall", "enrich10", "enrich5"]
SOURCES = [TEMPORAL, "random_holdout", "scaffold_cv"]
HEADLINE = ("trees4", "spearman")

N_BOOT = 10000
BOOT_SEED = 20260825


CONSERVATIVE_INFLATION = 1.222



def _prescribe(tab: pd.DataFrame, *, pool: list[str], metric: str,
               source: str, held: str) -> tuple[str, int]:
    """Documentation."""
    train = tab[(tab["protocol"] == source) & (tab["target"] != held)]
    norm = []
    for _t, g in train.groupby("target"):
        s = g.set_index("model")[metric].reindex(pool).dropna()
        if len(s) < 2 or not np.isfinite(s.max() - s.min()) or s.max() == s.min():
            continue
        norm.append((s - s.min()) / (s.max() - s.min()))
    if not norm:
        return "", 0
    mean_norm = pd.concat(norm, axis=1).mean(axis=1).reindex(pool).dropna()
    best = mean_norm.max()
    ties = [m for m in pool if m in mean_norm.index
            and np.isclose(mean_norm[m], best, atol=1e-12)]
    return ties[0], len(ties)


def _selfpick(tab: pd.DataFrame, *, pool: list[str], metric: str,
              target: str) -> tuple[str, int]:
    """Documentation."""
    s = (tab[(tab["protocol"] == "random_holdout") & (tab["target"] == target)]
         .set_index("model")[metric].reindex(pool).dropna())
    if s.empty:
        return "", 0
    best = s.max()
    ties = [m for m in pool if m in s.index and np.isclose(s[m], best, atol=1e-12)]
    return ties[0], len(ties)


def _temporal_scores(tab: pd.DataFrame, *, pool: list[str], metric: str,
                     target: str) -> pd.Series:
    return (tab[(tab["protocol"] == TEMPORAL) & (tab["target"] == target)]
            .set_index("model")[metric].reindex(pool).dropna())


def build_loto(tab: pd.DataFrame, *, pool_name: str, pool: list[str]) -> pd.DataFrame:
    """Documentation."""
    targets = sorted(tab["target"].unique())
    rows = []
    for metric in METRICS:
        for target in targets:
            tmp = _temporal_scores(tab, pool=pool, metric=metric, target=target)
            if len(tmp) < 2:
                continue
            tbest_model = str(tmp.idxmax())
            tbest = float(tmp.max())
            
            random_pick = float(np.mean([tbest - v for v in tmp.to_numpy()]))

            def _row(selector: str, chosen: str, n_ties: int) -> dict | None:
                if not chosen or chosen not in tmp.index:
                    return None
                return {"pool": pool_name, "metric": metric, "selector": selector,
                        "target": target, "prescribed": chosen,
                        "prescribed_temporal": float(tmp[chosen]),
                        "temporal_best_model": tbest_model, "temporal_best": tbest,
                        "regret": float(tbest - tmp[chosen]),
                        "random_pick_regret": random_pick, "n_ties": n_ties}

            for source in SOURCES:
                m, nt = _prescribe(tab, pool=pool, metric=metric,
                                   source=source, held=target)
                r = _row(f"loto_{source}", m, nt)
                if r:
                    rows.append(r)

            m, nt = _selfpick(tab, pool=pool, metric=metric, target=target)
            r = _row("cv_selfpick", m, nt)
            if r:
                rows.append(r)

            
            rows.append({"pool": pool_name, "metric": metric,
                         "selector": "random_pick", "target": target,
                         "prescribed": "", "prescribed_temporal": np.nan,
                         "temporal_best_model": tbest_model, "temporal_best": tbest,
                         "regret": random_pick, "random_pick_regret": random_pick,
                         "n_ties": len(tmp)})
    return pd.DataFrame(rows)



def summarize(lo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, metric, sel), g in lo.groupby(["pool", "metric", "selector"]):
        v = g["regret"].to_numpy()
        presc = g["prescribed"]
        modal = presc[presc != ""].value_counts()
        rows.append({
            "pool": pool, "metric": metric, "selector": sel, "n_targets": len(g),
            "prescribed_modal": (modal.index[0] if len(modal) else ""),
            "prescribed_modal_n": (int(modal.iloc[0]) if len(modal) else 0),
            "n_distinct_prescribed": int(presc[presc != ""].nunique()),
            "regret_mean": float(np.mean(v)), "regret_median": float(np.median(v)),
            "regret_p90": float(np.percentile(v, 90)), "regret_max": float(np.max(v)),
            "frac_zero": float(np.mean(v <= 1e-12)),
            "n_tie_prescription": int((g["n_ties"] > 1).sum()),
            "is_headline": (pool, metric) == HEADLINE,
        })
    return pd.DataFrame(rows).sort_values(
        ["pool", "metric", "selector"]).reset_index(drop=True)




COMPARISONS = [
    ("H1", "loto_random_holdout", "cv_selfpick",
     "CV vs CV"),
    ("H2", "loto_random_holdout", "random_pick",
     "CV vs random"),
    ("H3", "loto_random_holdout", "loto_temporal",
     "CV vs targetyear"),
    ("H4", "loto_temporal", "cv_selfpick",
     "vs CV"),
]
CONTAMINATED = {"H4"}


def _boot_ci(delta: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Documentation."""
    n = len(delta)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    mean = float(np.mean(delta))
    half = (hi - lo) / 2.0
    cons_half = half * CONSERVATIVE_INFLATION
    return {
        "delta_mean": mean,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_lo_conservative": mean - cons_half,
        "ci_hi_conservative": mean + cons_half,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "excludes_zero_conservative": bool((mean - cons_half) > 0
                                           or (mean + cons_half) < 0),
    }


def run_tests(lo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOT_SEED)
    rows, need = [], []
    for pool in ("trees4", "all7"):
        for metric in METRICS:
            sub = lo[(lo["pool"] == pool) & (lo["metric"] == metric)]
            wide = sub.pivot_table(index="target", columns="selector",
                                   values="regret")
            for hid, a, b in [(c[0], c[1], c[2]) for c in COMPARISONS]:
                if a not in wide.columns or b not in wide.columns:
                    continue
                pair = wide[[a, b]].dropna()
                
                delta = (pair[b] - pair[a]).to_numpy()
                n = len(delta)
                n_a_better = int((delta > 1e-12).sum())
                n_b_better = int((delta < -1e-12).sum())
                n_tie = int(n - n_a_better - n_b_better)
                if n_a_better + n_b_better > 0:
                    
                    p = float(wilcoxon(pair[b], pair[a],
                                       zero_method="zsplit").pvalue)
                else:
                    p = 1.0
                ci = _boot_ci(delta, rng)
                sd = float(np.std(delta, ddof=1)) if n > 1 else np.nan
                t_need = (int(np.ceil((1.96 * sd / abs(ci["delta_mean"])) ** 2))
                          if np.isfinite(sd) and abs(ci["delta_mean"]) > 1e-12
                          else -1)
                rows.append({
                    "hypothesis": hid, "pool": pool, "metric": metric,
                    "arm_a": a, "arm_b": b, "n_targets": n,
                    "mean_a": float(pair[a].mean()), "mean_b": float(pair[b].mean()),
                    "n_a_better": n_a_better, "n_b_better": n_b_better,
                    "n_tie": n_tie,
                    "wilcoxon_p": p,
                    "wilcoxon_sig_005": bool(p < 0.05),
                    **ci,
                    "contaminated_post_hoc": hid in CONTAMINATED,
                    "is_headline": (pool, metric) == HEADLINE,
                })
                need.append({"hypothesis": hid, "pool": pool, "metric": metric,
                             "delta_mean": ci["delta_mean"], "sd_delta": sd,
                             "T_needed": t_need, "n_targets_now": n})
    return pd.DataFrame(rows), pd.DataFrame(need)


# ------------------------------------------------------------------ main
def main() -> int:
    if not IN_CSV.exists():
        raise SystemExit(f"input: {IN_CSV} run first e01_regret_analysis.py")
    prot = load_protocol()
    pool_all = list(prot.models)          

    tab = pd.read_csv(IN_CSV)
    targets = sorted(tab["target"].unique())
    print(f"[e14] input {IN_CSV.name}: {len(tab)} , target {len(targets)}")
    print(f"[e14] trees4 = {POOL_TREES}")
    print(f"[e14] all7   = {pool_all}")
    print(f"[e14] source {SOURCES} {TEMPORAL}")
    print(f"[e14] = {HEADLINE[0]} × {HEADLINE[1]} p \n")

    lo = pd.concat([build_loto(tab, pool_name="trees4", pool=POOL_TREES),
                    build_loto(tab, pool_name="all7", pool=pool_all)],
                   ignore_index=True)
    summ = summarize(lo)
    tests, need = run_tests(lo)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_table(lo, OUT_DIR / "loto_by_source.csv")
    save_table(summ, OUT_DIR / "summary_by_source.csv")
    save_table(tests, OUT_DIR / "paired_tests.csv", float_fmt="%.6g")
    save_table(need, OUT_DIR / "required_targets.csv", float_fmt="%.6g")

    
    print("=" * 78)
    print("temporal target")
    print("=" * 78)
    for pool in ("trees4", "all7"):
        for metric in ("spearman", "kendall"):
            tag = "★" if (pool, metric) == HEADLINE else ""
            print(f"\n--- {pool} × {metric}{tag} ---")
            s = summ[(summ["pool"] == pool) & (summ["metric"] == metric)]
            print(s[["selector", "prescribed_modal", "prescribed_modal_n",
                     "n_distinct_prescribed", "regret_mean", "regret_median",
                     "regret_p90", "regret_max", "frac_zero"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("Wilcoxon bootstrap CI §4.4")
    print("=" * 78)
    for hid, a, b, desc in COMPARISONS:
        print(f"\n### {hid}  {desc}")
        print(f"Δ_t = regret[{b}] − regret[{a}] ( >0 {a} )")
        t = tests[tests["hypothesis"] == hid]
        for _, r in t.iterrows():
            star = " ★" if r["is_headline"] else "  "
            cons = "0" if r["excludes_zero_conservative"] else "0"
            print(f"{star}{r['pool']:6s} {r['metric']:9s} "
                  f"{r['arm_a'][:22]:22s} {r['mean_a']:.4f} vs {r['mean_b']:.4f}  "
                  f"{r['n_a_better']:2d}/{r['n_b_better']:2d}/{r['n_tie']:2d}"
                  f"p={r['wilcoxon_p']:.4f}  "
                  f"Δ̄={r['delta_mean']:+.4f} "
                  f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                  f"[{r['ci_lo_conservative']:+.4f},"
                  f"{r['ci_hi_conservative']:+.4f}] {cons}")

    print("\n" + "=" * 78)
    print("target T_needed = ceil((1.96·sd(Δ)/|Δ̄|)²) §4.4.2")
    print("=" * 78)
    ok = need[need["T_needed"] > 0]
    print(f"{len(ok)} : {ok['T_needed'].median():.0f}"
          f"IQR {ok['T_needed'].quantile(.25):.0f}–{ok['T_needed'].quantile(.75):.0f}  "
          f"min {ok['T_needed'].min():.0f}  max {ok['T_needed'].max():.0f}  "
          f"{len(targets)}")
    for hid in ("H1", "H3"):
        h = ok[ok["hypothesis"] == hid]
        hh = h[(h["pool"] == HEADLINE[0]) & (h["metric"] == HEADLINE[1])]
        if len(hh):
            print(f"{hid} : Δ̄={hh['delta_mean'].iloc[0]:+.4f}"
                  f"sd={hh['sd_delta'].iloc[0]:.4f} "
                  f"→ T_needed = {hh['T_needed'].iloc[0]:.0f}")

    write_run_meta(
        OUT_DIR, target="__cross_target__", stage="e14_loto_source",
        protocol_fingerprint=prot.fingerprint,
        extra={
            "declared_in": DECLARED_IN,
            "input": str(IN_CSV.relative_to(ROOT)),
            "targets": targets,
            "pools": {"trees4": POOL_TREES, "all7": pool_all},
            "sources": SOURCES,
            "metrics": METRICS,
            "headline_cell": {"pool": HEADLINE[0], "metric": HEADLINE[1],
                              "rule": "23 p"},
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "ci_scope": ("target molecules input fold-mean"
                         "moleculesprediction generalization"
                         f"= × {CONSERVATIVE_INFLATION}"
                         "e13"),
            "wilcoxon_zero_method": "zsplit e13",
            "post_hoc_disclosure": (
                "H1′–H4′ bootstrap H1–H4 Wilcoxon"
                "message"
                "−0.935 ⇒ results"
                "H4 §2.3 contaminated_post_hoc=True"),
            "tail_no_test": "regret_p90 / regret_max p §4.4",
            "determinism": ("bootstrap seed RNG input"
                            "message"),
            "limitation": ("① 7 model trainingmodel"
                           "GNN ② target targets_manifest"
                           "③ metrics ④ bagging/boosting"
                           "bagging"),
            "red_lines": "note_8/17 §6 22–26",
        })
    print(f"\n[e14] -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
