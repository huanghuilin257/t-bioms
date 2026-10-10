#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()  

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from scipy.stats import binomtest                      # noqa: E402

from pparbench.config import load_protocol             # noqa: E402
from pparbench.io import save_table, write_run_meta    # noqa: E402

from e14_loto_source import (                          # noqa: E402
    METRICS, POOL_TREES, TEMPORAL, _prescribe, _selfpick,
    CONSERVATIVE_INFLATION, N_BOOT, BOOT_SEED,
)
from e01_regret_analysis import FROZEN20, _metrics     # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
IN_CSV = ROOT / "results" / "cross_target" / "e01_regret" / "e01_foldmean_all_metrics.csv"
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"

SEL_PRIMARY = "scaffold_cv"          
SEL_ROBUST = "random_holdout"        
SEL_PROTOCOLS = [SEL_PRIMARY, SEL_ROBUST]
HEADLINE = ("trees4", "spearman", SEL_PRIMARY)

EPS_REL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]                  
EPS_ABS = [0.005, 0.010, 0.020, 0.030, 0.050]             


HOLDOUT_EXCLUDED = {"drd3": 0.8547, "htr1a": 0.4296}                    
HOLDOUT_FLAGGED = {"htr6": 0.2970, "slc6a3": 0.2065, "adra1d": 0.2045}  
HOLDOUT_CLEAN = ["hrh4", "trpv1", "mtnr1a"]
HOLDOUT_MAIN = HOLDOUT_CLEAN + list(HOLDOUT_FLAGGED)                    



def per_fold_table(targets: list[str]) -> pd.DataFrame:
    """Documentation."""
    frames, missing = [], []
    for t in targets:
        p = ROOT / "results" / t / "p03_protocol_matrix" / "predictions.csv"
        if not p.exists():
            missing.append(t)
            continue
        pred = pd.read_csv(p)
        rows = []
        for (protocol, model, fold), g in pred.groupby(["protocol", "model", "fold"]):
            rows.append({"target": t, "protocol": protocol, "model": model,
                         "fold": fold, "n": len(g),
                         **_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())})
        frames.append(pd.DataFrame(rows))
    if missing:
        raise SystemExit(f"[e20] predictions.csv : {missing}")
    return pd.concat(frames, ignore_index=True)


def _check_foldmean(fold: pd.DataFrame, foldmean: pd.DataFrame) -> None:
    """Documentation."""
    worst = {}
    for m in METRICS:
        got = (fold.groupby(["target", "protocol", "model"])[m].mean()
               .rename("recomputed").reset_index())
        mm = foldmean[["target", "protocol", "model", m]].merge(
            got, on=["target", "protocol", "model"], how="inner")
        diff = (mm[m] - mm["recomputed"]).abs()
        worst[m] = float(diff.max())
        bad = mm[diff > 5e-5]
        if len(bad):
            print(bad.head(10).to_string(index=False))
            raise SystemExit(f"[e20] {m} e01 inconsistent {len(bad)}/{len(mm)}")
    print("[e20] :"
          + ", ".join(f"{k}={v:.2e}" for k, v in worst.items()))


def epsilon_rows(fold: pd.DataFrame, foldmean: pd.DataFrame, *,
                 pool_name: str, pool: list[str]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for metric in METRICS:
        for sel in SEL_PROTOCOLS:
            for t in sorted(foldmean["target"].unique()):
                s = (foldmean[(foldmean.protocol == sel) & (foldmean.target == t)]
                     .set_index("model")[metric].reindex(pool).dropna())
                if len(s) < 2:
                    continue
                order = s.sort_values(ascending=False)
                top1, top2 = order.index[0], order.index[1]
                gap = float(order.iloc[0] - order.iloc[1])

                w = fold[(fold.protocol == sel) & (fold.target == t)]
                a = w[w.model == top1].set_index("fold")[metric]
                b = w[w.model == top2].set_index("fold")[metric]
                common = a.index.intersection(b.index)          
                d = (a.loc[common] - b.loc[common]).astype(float)
                k = int(len(d))
                se = float(d.std(ddof=1) / np.sqrt(k)) if k >= 2 else np.nan

                rows.append({
                    "pool": pool_name, "metric": metric, "sel_protocol": sel,
                    "target": t, "top1": top1, "top2": top2, "gap": round(gap, 6),
                    "k_folds": k, "se_t": se if np.isnan(se) else round(se, 6),
                    "foldmean_top1": round(float(order.iloc[0]), 6),
                    "n_models": int(len(s)),
                })
    return pd.DataFrame(rows)


def eps0_table(eps: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    g = eps.dropna(subset=["se_t"]).groupby(["pool", "metric", "sel_protocol"])["se_t"]
    out = g.agg(eps0="median", se_mean="mean", se_p25=lambda s: s.quantile(.25),
                se_p75=lambda s: s.quantile(.75), n_targets="size").reset_index()
    return out.round(6)


def tied_set(foldmean: pd.DataFrame, *, pool: list[str], metric: str,
             sel: str, target: str, eps: float) -> list[str]:
    """Documentation."""
    s = (foldmean[(foldmean.protocol == sel) & (foldmean.target == target)]
         .set_index("model")[metric].reindex(pool).dropna())
    if s.empty:
        return []
    best = s.max()
    return [m for m in pool if m in s.index and (best - s[m]) <= eps + 1e-12]


def event_counts(eps: pd.DataFrame, foldmean: pd.DataFrame, e0: pd.DataFrame,
                 pools: dict[str, list[str]]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    key = e0.set_index(["pool", "metric", "sel_protocol"])["eps0"].to_dict()
    for (pool_name, metric, sel), g in eps.groupby(["pool", "metric", "sel_protocol"]):
        e0v = key.get((pool_name, metric, sel), np.nan)
        grid = ([("rel", f"{r}xeps0", r * e0v) for r in EPS_REL]
                + [("abs", f"{a:g}", a) for a in EPS_ABS])
        for kind, label, val in grid:
            if not np.isfinite(val):
                continue
            ev = g[g.gap <= val + 1e-12]
            sizes = [len(tied_set(foldmean, pool=pools[pool_name], metric=metric,
                                  sel=sel, target=t, eps=val)) for t in ev.target]
            rows.append({
                "pool": pool_name, "metric": metric, "sel_protocol": sel,
                "eps_kind": kind, "eps_label": label, "eps_value": round(val, 6),
                "n_event": int(len(ev)), "n_total": int(len(g)),
                "frac_event": round(len(ev) / len(g), 4) if len(g) else np.nan,
                "mean_tied_size": round(float(np.mean(sizes)), 3) if sizes else np.nan,
                "max_tied_size": int(max(sizes)) if sizes else 0,
                "n_tied_ge2": int(sum(1 for x in sizes if x >= 2)),
                "is_headline": (pool_name, metric, sel) == HEADLINE and label == "1.0xeps0",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ S2
def loto_ranking(tab: pd.DataFrame, *, pool: list[str], metric: str,
                 source: str, held: str) -> pd.Series:
    """Documentation."""
    train = tab[(tab["protocol"] == source) & (tab["target"] != held)]
    norm = []
    for _t, g in train.groupby("target"):
        s = g.set_index("model")[metric].reindex(pool).dropna()
        if len(s) < 2 or not np.isfinite(s.max() - s.min()) or s.max() == s.min():
            continue
        norm.append((s - s.min()) / (s.max() - s.min()))
    if not norm:
        return pd.Series(dtype=float)
    mean_norm = pd.concat(norm, axis=1).mean(axis=1).reindex(pool).dropna()
    
    ranking = mean_norm.sort_values(ascending=False, kind="stable")
    ref, _ = _prescribe(tab, pool=pool, metric=metric, source=source, held=held)
    if ref and ranking.index[0] != ref:
        raise RuntimeError(f"LOTO e14 _prescribe inconsistent: {ranking.index[0]} vs {ref}")
    return ranking


def s2_rows(tab_eval: pd.DataFrame, tab_src: pd.DataFrame, e0: pd.DataFrame,
            pools: dict[str, list[str]], *, targets: list[str], loto: bool,
            eps_labels: list[str]) -> pd.DataFrame:
    """Documentation."""
    key = e0.set_index(["pool", "metric", "sel_protocol"])["eps0"].to_dict()
    rel = {f"{r}xeps0": r for r in EPS_REL}
    rows = []
    for pool_name, pool in pools.items():
        for metric in METRICS:
            for sel in SEL_PROTOCOLS:
                e0v = key.get((pool_name, metric, sel), np.nan)
                for label in eps_labels:
                    val = rel[label] * e0v if label in rel else float(label)
                    if not np.isfinite(val):
                        continue
                    for t in targets:
                        tied = tied_set(tab_eval, pool=pool, metric=metric,
                                        sel=sel, target=t, eps=val)
                        if len(tied) < 2:      
                            continue
                        temp = (tab_eval[(tab_eval.protocol == TEMPORAL)
                                         & (tab_eval.target == t)]
                                .set_index("model")[metric].reindex(pool).dropna())
                        if temp.empty:
                            continue
                        best = float(temp.max())
                        reg = {m: best - float(temp[m]) for m in temp.index}
                        avail = [m for m in tied if m in reg]
                        if not avail:
                            continue
                        coin = float(np.mean([reg[m] for m in avail]))   

                        rank = loto_ranking(tab_src, pool=pool, metric=metric,
                                            source=sel, held=(t if loto else ""))
                        if rank.empty:
                            continue
                        uncon = str(rank.index[0])
                        inside = [m for m in rank.index if m in tied]
                        con = str(inside[0]) if inside else uncon
                        cv_pick, _ = _selfpick(tab_eval, pool=pool, metric=metric,
                                               target=t)
                        rows.append({
                            "pool": pool_name, "metric": metric, "sel_protocol": sel,
                            "eps_label": label, "eps_value": round(val, 6),
                            "target": t, "n_tied": len(tied),
                            "tied_set": "|".join(tied),
                            "coinflip_regret": round(coin, 6),
                            "constrained_model": con,
                            "constrained_regret": round(reg.get(con, np.nan), 6),
                            "unconstrained_model": uncon,
                            "unconstrained_regret": round(reg.get(uncon, np.nan), 6),
                            "cv_selfpick_model": cv_pick,
                            "cv_selfpick_regret": round(reg.get(cv_pick, np.nan), 6),
                            "delta_constrained": round(coin - reg.get(con, np.nan), 6),
                            "delta_unconstrained": round(coin - reg.get(uncon, np.nan), 6),
                            "is_headline": (pool_name, metric, sel) == HEADLINE,
                        })
    return pd.DataFrame(rows)



def _boot_ci(vals: np.ndarray, units: np.ndarray | None, rng) -> dict:
    """Documentation."""
    if units is None:
        idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        means = vals[idx].mean(axis=1)
    else:
        groups = {u: np.where(units == u)[0] for u in np.unique(units)}
        keys = list(groups)
        means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            sel = np.concatenate([groups[keys[i]] for i in pick])
            means[b] = vals[sel].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    m = float(vals.mean())
    half = (hi - lo) / 2 * CONSERVATIVE_INFLATION
    return {"delta_mean": m, "ci_lo": float(lo), "ci_hi": float(hi),
            "ci_lo_cons": m - half, "ci_hi_cons": m + half,
            "n_units": len(np.unique(units)) if units is not None else len(vals)}


def run_tests(s2: pd.DataFrame, clusters: pd.DataFrame, *,
              significance: bool) -> pd.DataFrame:
    """Documentation."""
    cl = clusters.set_index("slug")
    rows = []
    for (pool, metric, sel, label), g in s2.groupby(
            ["pool", "metric", "sel_protocol", "eps_label"]):
        for variant in ("constrained", "unconstrained"):
            sub = g[np.isfinite(g[f"delta_{variant}"].to_numpy(dtype=float))]
            v = sub[f"delta_{variant}"].to_numpy(dtype=float)
            if len(v) < 2:
                continue
            sd = float(v.std(ddof=1))
            mean = float(v.mean())
            base = {"pool": pool, "metric": metric, "sel_protocol": sel,
                    "eps_label": label, "variant": variant, "n_targets": len(v),
                    "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum()),
                    "n_zero": int((v == 0).sum()), "sd_delta": sd,
                    "T_needed": (int(np.ceil((1.96 * sd / mean) ** 2))
                                 if mean != 0 else np.nan),
                    "is_headline": (pool, metric, sel) == HEADLINE}
            if significance:
                nz = int((v > 0).sum() + (v < 0).sum())
                base["binom_p"] = (float(binomtest(int((v > 0).sum()), nz, 0.5).pvalue)
                                   if nz else np.nan)
            else:
                base["binom_p"] = np.nan       
            for unit_name, col in (("target", None), ("cluster_fine", "cluster_fine"),
                                   ("cluster_coarse", "cluster_coarse")):
                units = None
                if col is not None:
                    units = np.array([cl.loc[t, col] if t in cl.index else t
                                      for t in sub["target"]])
                ci = _boot_ci(v, units, np.random.default_rng(BOOT_SEED))
                rows.append({**base, "boot_unit": unit_name, **ci})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["epsilon", "full"], default="epsilon")
    ap.add_argument("--holdout", action="store_true",
                    help="note_8/24 §1.4 6")
    args = ap.parse_args()

    if not IN_CSV.exists():
        raise SystemExit(f"input: {IN_CSV} run first e01_regret_analysis.py")
    prot = load_protocol()
    pools = {"trees4": POOL_TREES, "all7": list(prot.models)}

    tab_all = pd.read_csv(IN_CSV)
    frozen_present = [t for t in FROZEN20 if t in set(tab_all["target"])]
    if args.holdout:
        targets = [t for t in HOLDOUT_MAIN if t in set(tab_all["target"])]
        out_dir = ROOT / "results" / "cross_target" / "e20_two_stage_holdout"
    else:
        targets = frozen_present
        out_dir = ROOT / "results" / "cross_target" / "e20_two_stage"
    tab = tab_all[tab_all["target"].isin(targets)].copy()
    
    tab_src = tab_all[tab_all["target"].isin(frozen_present)].copy()
    print(f"[e20] {'' if args.holdout else 'FROZEN20'}: {len(targets)} {targets}")
    print(f"[e20] source: FROZEN20 {len(frozen_present)}"
          f"{' LOTO ' if not args.holdout else ' '}")
    if args.holdout:
        print(f"[e20] note_8/24 §1.4 : {HOLDOUT_EXCLUDED}")
        print(f"[e20] : {HOLDOUT_FLAGGED}")
        print(f"[e20] {len(HOLDOUT_CLEAN)} : {HOLDOUT_CLEAN}")
        print("[e20] 88")
    else:
        print("[e20] 84 FROZEN20 S2 p")

    fold = per_fold_table(targets)
    _check_foldmean(fold, tab)
    eps = pd.concat([epsilon_rows(fold, tab, pool_name=k, pool=v)
                     for k, v in pools.items()], ignore_index=True)
    e0 = eps0_table(eps)
    ev = event_counts(eps, tab, e0, pools)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(eps, out_dir / "e20_epsilon_by_target.csv", float_fmt="%.6g")
    save_table(e0, out_dir / "e20_epsilon0.csv", float_fmt="%.6g")
    save_table(ev, out_dir / "e20_events.csv", float_fmt="%.6g")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print("ε₀ = median_t(SE_t) SE_t = sd()/sqrt(k_t) §3.1")
    print("=" * 78)
    print(e0.to_string(index=False))
    h = ev[(ev.pool == HEADLINE[0]) & (ev.metric == HEADLINE[1])
           & (ev.sel_protocol == HEADLINE[2])]
    print(f"\n {HEADLINE} ε 85")
    print(h.to_string(index=False))

    if args.step == "epsilon":
        write_meta(out_dir, prot, args, targets)
        print("\n[e20] --step epsilon ε₀ §7 3"
              "--step full")
        return 0

    
    labels = [f"{r}xeps0" for r in EPS_REL] + [f"{a:g}" for a in EPS_ABS]
    s2 = s2_rows(tab, tab_src, e0, pools, targets=targets,
                 loto=not args.holdout, eps_labels=labels)
    if s2.empty:
        raise SystemExit("[e20] no e20_events.csv")
    clusters = pd.read_csv(CLUSTERS, comment="#")
    tests = run_tests(s2, clusters, significance=not args.holdout)

    save_table(s2, out_dir / "e20_s2_by_target.csv", float_fmt="%.6g")
    save_table(tests, out_dir / "e20_s2_tests.csv", float_fmt="%.6g")

    print("\n" + "=" * 78)
    print("S2 Δ = − =")
    print("93 target/ inconsistent cluster_coarse 102")
    print("=" * 78)
    hs = tests[(tests.is_headline) & (tests.eps_label == "1.0xeps0")]
    cols = ["variant", "boot_unit", "n_targets", "n_units", "n_pos", "n_neg", "n_zero",
            "delta_mean", "ci_lo_cons", "ci_hi_cons", "binom_p", "T_needed"]
    print(hs[cols].to_string(index=False))
    print("\n ε=ε₀")
    hg = s2[(s2.is_headline) & (s2.eps_label == "1.0xeps0")]
    print(hg[["target", "n_tied", "tied_set", "coinflip_regret", "constrained_model",
              "constrained_regret", "delta_constrained",
              "unconstrained_model", "delta_unconstrained"]].to_string(index=False))
    write_meta(out_dir, prot, args, targets)
    return 0


def write_meta(out_dir: Path, prot, args, targets: list[str]) -> None:
    write_run_meta(out_dir, target="cross_target",
                   stage="e20_two_stage" + ("_holdout" if args.holdout else ""),
                   protocol_fingerprint=prot.fingerprint,
                   extra={"declared_in": DECLARED_IN, "step": args.step,
                          "targets": targets, "eps_rel": EPS_REL, "eps_abs": EPS_ABS,
                          "headline": list(HEADLINE),
                          "holdout_excluded": HOLDOUT_EXCLUDED,
                          "holdout_flagged": HOLDOUT_FLAGGED,
                          "holdout_clean": HOLDOUT_CLEAN,
                          "post_hoc_no_pvalue_evidence": not args.holdout,
                          "no_significance_testing": bool(args.holdout)})


if __name__ == "__main__":
    raise SystemExit(main())
