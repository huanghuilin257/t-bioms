#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from pparbench.threads import apply_thread_limits              # noqa: E402

N_THREADS = apply_thread_limits()

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
from scipy.stats import binomtest                              # noqa: E402

from pparbench.io import save_table, write_run_meta            # noqa: E402
from e17_r2_axis import FROZEN20                               # noqa: E402


from e20_two_stage import _boot_ci                             # noqa: E402
from e14_loto_source import BOOT_SEED, N_BOOT                  # noqa: E402


from e04_hparam_pool import _pick                              # noqa: E402



from e01_regret_analysis import _metrics                        # noqa: E402




from pparbench.config import load_target as load_target_cfg     # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
SELECTORS = ["random_holdout", "scaffold_cv"]
EVAL_PROTOCOL = "temporal"
METRICS = ["spearman", "kendall"]        
K_MAIN = 2                               
K_SECONDARY = 4                          
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"
BOOT_UNITS = [("target", None), ("cluster_fine", "cluster_fine"),
              ("cluster_coarse", "cluster_coarse")]
OUTDIR = ROOT / "results" / "cross_target" / "e04_level_dissociation"


TREE_FAMILIES = {"boosting": ["lightgbm", "hgb"], "bagging": ["rf", "extratrees"]}
GNN_ALGOS = ["dmpnn", "attentivefp", "gin"]
GNN_SEEDS = [0, 1, 2]                    
K_CONFIGS = 8                            
GNN_SPEARMAN_TOL = 1e-9                  




FLOAT_PRECISION = "round_trip"



E04G_DIR = "e04g_hparam_pool_gnn"
N_FOLDS_CV = {"random_holdout": 3, "scaffold_cv": 5}   
CELLS_PER_FOLD = len(GNN_ALGOS) * K_CONFIGS * len(GNN_SEEDS)   # 3×8×3 = 72



def _expected_rows(slug: str) -> tuple[int, int]:
    """Documentation."""
    cps = load_target_cfg(slug).cutpoints or []
    if not cps:
        raise RuntimeError(f"⛔ {slug} cutpoints are missing from configs/targets/{slug}.yaml"
                           f"⛔ protocol.yaml default_cutpoints")
    return (sum(N_FOLDS_CV.values()) + len(cps)) * CELLS_PER_FOLD, len(cps)




def e04g_complete(verbose: bool = False) -> tuple[bool, list[str], dict]:
    """Documentation."""
    missing, detail = [], {}
    n_collapsed: dict[str, int] = {}      
    for t in FROZEN20:
        p = ROOT / "results" / t / E04G_DIR / "per_fold.csv"
        if not p.exists():
            missing.append(t); detail[t] = "message"
            continue
        d = pd.read_csv(p, float_precision=FLOAT_PRECISION)
        problems = []
        want_rows, n_cp = _expected_rows(t)
        if len(d) != want_rows:
            problems.append(f"{len(d)}/{want_rows} {n_cp}")
        for prot, nf in {**N_FOLDS_CV, "temporal": n_cp}.items():
            sub = d[d.protocol == prot]
            want = nf * CELLS_PER_FOLD
            if len(sub) != want:
                problems.append(f"{prot} {len(sub)}/{want}")
        got_algo = set(d["algo"].unique()) if "algo" in d.columns else set()
        if got_algo != set(GNN_ALGOS):
            problems.append(f"algo {sorted(got_algo)}")
        got_seed = set(d["seed"].unique()) if "seed" in d.columns else set()
        if got_seed != set(GNN_SEEDS):
            problems.append(f"seed {sorted(got_seed)}")
        
        
        
        
        
        
        
        
        nan_rho = (d["spearman"].isna() if "spearman" in d.columns
                   else pd.Series(False, index=d.index))
        has_cols = {"pred_std", "n_pred_nan"} <= set(d.columns)
        collapsed = ((d["pred_std"] == 0) & (d["n_pred_nan"] == 0) if has_cols
                     else pd.Series(False, index=d.index))
        if not has_cols:
            problems.append("pred_std/n_pred_nan ⇒")
        n_nan_bad = int((nan_rho & ~collapsed).sum())
        if n_nan_bad:
            problems.append(f"NaN spearman {n_nan_bad}")
        n_collapsed[t] = int(collapsed.sum())
        if problems:
            missing.append(t); detail[t] = "; ".join(problems)
    if verbose and detail:
        for t, why in detail.items():
            print(f"[e04ld]     {t:9s} {why}")
    if verbose and n_collapsed:
        tot = sum(n_collapsed.values())
        top = sorted(n_collapsed.items(), key=lambda kv: -kv[1])[:3]
        print(f"[e04ld] 3 0 ⛔ {tot}"
              f"{top}")
    detail["__n_collapsed__"] = n_collapsed
    return (len(missing) == 0), missing, detail


def load_target(slug: str) -> pd.DataFrame:
    p = ROOT / "results" / slug / "e04_hparam_pool" / "per_fold.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, float_precision=FLOAT_PRECISION)


def impute_collapse(d: pd.DataFrame) -> tuple[pd.DataFrame, int, list]:
    """Documentation."""
    d = d.copy()
    n_imp, unexplained = 0, []
    for m in METRICS:
        if m not in d.columns:
            continue
        bad = d[m].isna()
        if not bad.any():
            continue
        collapsed = bad & (d.get("pred_std", pd.Series(1.0, index=d.index)) == 0.0)\
                        & (d.get("n_pred_nan", pd.Series(0, index=d.index)) == 0)
        d.loc[collapsed, m] = 0.0
        n_imp += int(collapsed.sum())
        rest = bad & ~collapsed
        for r in d.loc[rest].itertuples():
            unexplained.append((m, r.model, r.protocol, getattr(r, "fold", "?"),
                                getattr(r, "seed", "?")))
    return d, n_imp, unexplained


def load_gnn(slug: str) -> tuple[pd.DataFrame, dict]:
    """Documentation."""
    d = ROOT / "results" / slug / E04G_DIR
    pf = pd.read_csv(d / "per_fold.csv", float_precision=FLOAT_PRECISION)
    pred = pd.read_csv(d / "predictions.csv", float_precision=FLOAT_PRECISION)
    keys = ["protocol", "fold", "model", "seed"]
    rows = []
    for kv, g in pred.groupby(keys, sort=False):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        ok = np.isfinite(yp)          
        m = _metrics(yt[ok], yp[ok])
        rows.append(dict(zip(keys, kv), **{k: float(m[k]) for k in METRICS}))
    rec = pd.DataFrame(rows)

    stored = pf[keys + ["spearman"]].rename(columns={"spearman": "spearman__stored"})
    merged = pf.drop(columns=[c for c in METRICS if c in pf.columns]).merge(
        rec, on=keys, how="left", validate="one_to_one").merge(
        stored, on=keys, how="left", validate="one_to_one")

    orphan = merged[merged["spearman"].isna() & merged["spearman__stored"].notna()]
    if len(orphan):
        raise RuntimeError(
            f"⛔ {slug}: {len(orphan)} per_fold.csv predictions.csv"
            f"predictions.csv ？ \n{orphan.head(3)}")
    nan_mismatch = merged[merged["spearman"].isna() != merged["spearman__stored"].isna()]
    if len(nan_mismatch):
        raise RuntimeError(
            f"⛔ {slug}: {len(nan_mismatch)} NaN inconsistent ⇒"
            f"3 0 \n{nan_mismatch.head(3)}")
    both = merged["spearman"].notna()
    dev = float((merged.loc[both, "spearman"]
                 - merged.loc[both, "spearman__stored"]).abs().max()) if both.any() else 0.0
    if dev > GNN_SPEARMAN_TOL:
        raise RuntimeError(
            f"⛔ {slug}: spearman {dev:.3e} > {GNN_SPEARMAN_TOL:.0e} ⇒"
            f"** kendall ** 4-A ②")
    diag = {"n_cells": len(merged), "spearman_max_dev": dev,
            "n_constant_cells": int((~both).sum())}
    return merged.drop(columns=["spearman__stored"]), diag


def build_agg(slug: str, *, include_gnn: bool, gnn_agg: str,
              exclude_gin: bool) -> tuple[pd.DataFrame, dict]:
    """Documentation."""
    keys = ["protocol", "model", "algo", "family"]
    trees = load_target(slug)
    trees, n_imp_t, unexp_t = impute_collapse(trees)
    agg = trees.groupby(keys)[METRICS].mean().reset_index()
    diag = {"n_collapsed_imputed": n_imp_t, "n_nan_unexplained": len(unexp_t),
            "unexplained": unexp_t, "gnn_used": False}
    if not include_gnn:
        return agg, diag

    g, gnn_recalc = load_gnn(slug)      
    if exclude_gin:
        g = g[g.algo != "gin"]
    g, n_imp_g, unexp_g = impute_collapse(g)
    if gnn_agg == "seed_median":
        per_fold = g.groupby(keys + ["fold"])[METRICS].median().reset_index()
        gagg = per_fold.groupby(keys)[METRICS].mean().reset_index()
    elif gnn_agg == "fold_mean_only":
        gagg = g.groupby(keys)[METRICS].mean().reset_index()
    else:
        raise ValueError(gnn_agg)
    diag.update({"n_collapsed_imputed": n_imp_t + n_imp_g,
                 "n_nan_unexplained": len(unexp_t) + len(unexp_g),
                 "unexplained": unexp_t + unexp_g, "gnn_used": True,
                 "gnn_agg": gnn_agg, "exclude_gin": exclude_gin,
                 "gnn_kendall_recalc": gnn_recalc})
    return pd.concat([agg, gagg], ignore_index=True), diag




def subpool_stats(temporal: pd.Series, sel: pd.Series,
                  pool: list[str], k: int) -> dict:
    """Documentation."""
    if len(pool) < k:
        return {}
    regrets, floors, ties, spreads = [], [], [], []
    for q in itertools.combinations(pool, k):
        q = list(q)
        tv = temporal[q]
        t_best = float(tv.max())
        chosen, n_ties = _pick(sel, q)
        if not chosen:
            continue
        regrets.append(t_best - float(temporal[chosen]))
        floors.append(t_best - float(tv.mean()))
        ties.append(n_ties)
        spreads.append(float(tv.max() - tv.min()))
    if not regrets:
        return {}
    return {
        "n_subpools": len(regrets),
        "mean_regret": float(np.mean(regrets)),
        "mean_floor": float(np.mean(floors)),
        "mean_spread": float(np.mean(spreads)),
        "frac_zero_regret": float(np.mean([r <= 0 for r in regrets])),
        "mean_n_ties": float(np.mean(ties)),
        "frac_tied": float(np.mean([n > 1 for n in ties])),
    }


def build_pools(agg_sel: pd.Series, algo_of: dict, fam_of: dict,
                algos: list[str], families: dict,
                with_l3: bool) -> dict:
    """Documentation."""
    pools = {}
    l1_pools = {}
    for a in algos:
        members = [m for m in agg_sel.index if algo_of.get(m) == a]
        if members:
            l1_pools[a] = sorted(members)
    pools["L1"] = l1_pools                       

    l2_members = {}
    for a, members in l1_pools.items():
        champ, _ = _pick(agg_sel, members)
        if champ:
            l2_members[a] = champ
    pools["L2"] = {"all_algos": sorted(l2_members.values())}

    if with_l3:
        l3_members = []
        for fam, fam_algos in families.items():
            cand = [l2_members[a] for a in fam_algos if a in l2_members]
            champ, _ = _pick(agg_sel, cand)
            if champ:
                l3_members.append(champ)
        pools["L3"] = {"all_families": sorted(l3_members)}
    return pools


def analyse_target(slug: str, agg: pd.DataFrame, *,
                   algos: list[str], families: dict,
                   with_l3: bool) -> list[dict]:
    algo_of = dict(zip(agg["model"], agg["algo"]))
    fam_of = dict(zip(agg["model"], agg["family"]))
    rows = []
    for metric in METRICS:
        piv = agg.pivot_table(index="model", columns="protocol", values=metric)
        if EVAL_PROTOCOL not in piv.columns:
            continue
        temporal = piv[EVAL_PROTOCOL].dropna()
        for selector in SELECTORS:
            if selector not in piv.columns:
                continue
            sel = piv[selector].dropna()
            pools = build_pools(sel, algo_of, fam_of, algos, families, with_l3)
            for level, named in pools.items():
                for k in (K_MAIN, K_SECONDARY):
                    
                    per_pool = []
                    for pname, members in named.items():
                        members = [m for m in members if m in temporal.index]
                        st = subpool_stats(temporal, sel, members, k)
                        if st:
                            st["pool_name"] = pname
                            st["native_pool_size"] = len(members)
                            per_pool.append(st)
                    if not per_pool:
                        continue
                    d = pd.DataFrame(per_pool)
                    w = d["n_subpools"].sum()
                    rows.append({
                        "target": slug, "metric": metric, "selector": selector,
                        "level": level, "k": k,
                        "n_pools": len(d),
                        "n_subpools_total": int(w),
                        "native_pool_size": float(d["native_pool_size"].mean()),
                        
                        
                        "mean_regret": float(d["mean_regret"].mean()),
                        "mean_floor": float(d["mean_floor"].mean()),
                        "mean_spread": float(d["mean_spread"].mean()),
                        "regret_over_floor": (float(d["mean_regret"].mean()
                                                    / d["mean_floor"].mean())
                                              if d["mean_floor"].mean() > 0
                                              else np.nan),
                        "frac_zero_regret": float(d["frac_zero_regret"].mean()),
                        "mean_n_ties": float(d["mean_n_ties"].mean()),
                        "frac_tied": float(d["frac_tied"].mean()),
                    })
    return rows




def widest_ci(values: np.ndarray, targets: list[str], cl: pd.DataFrame) -> dict:
    """Documentation."""
    out = {}
    widest = None
    for unit_name, col in BOOT_UNITS:
        units = (None if col is None
                 else np.array([cl.loc[t, col] for t in targets]))
        ci = _boot_ci(values, units, np.random.default_rng(BOOT_SEED))
        w = ci["ci_hi"] - ci["ci_lo"]
        out[f"ci_lo__{unit_name}"] = ci["ci_lo"]
        out[f"ci_hi__{unit_name}"] = ci["ci_hi"]
        out[f"n_units__{unit_name}"] = ci["n_units"]
        if widest is None or w > widest[0]:
            widest = (w, unit_name, ci)
    _, name, ci = widest
    out.update({"boot_unit_widest": name, "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"], "ci_width": ci["ci_hi"] - ci["ci_lo"]})
    return out


def summarise(per_target: pd.DataFrame,
              cl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for (metric, selector, level, k), g in per_target.groupby(
            ["metric", "selector", "level", "k"]):
        g = g.sort_values("target")
        for col in ("mean_regret", "mean_floor", "regret_over_floor"):
            keep = np.isfinite(g[col].to_numpy(dtype=float))
            v = g.loc[keep, col].to_numpy(dtype=float)
            ts = g.loc[keep, "target"].tolist()
            if len(v) < 2:
                continue
            row = {"metric": metric, "selector": selector, "level": level,
                   "k": k, "quantity": col, "n_targets": len(v),
                   "mean": float(v.mean())}
            row.update(widest_ci(v, ts, cl))
            out.append(row)
    summary = pd.DataFrame(out)

    
    signs = []
    levels = sorted(per_target["level"].unique())
    for (metric, selector, k), g in per_target.groupby(
            ["metric", "selector", "k"]):
        piv = g.pivot_table(index="target", columns="level",
                            values=["mean_regret", "mean_floor"])
        for lo_lv, hi_lv in itertools.combinations(levels, 2):
            for quantity in ("mean_regret", "mean_floor"):
                if (quantity, lo_lv) not in piv.columns: continue
                if (quantity, hi_lv) not in piv.columns: continue
                d = (piv[(quantity, hi_lv)] - piv[(quantity, lo_lv)]).dropna()
                if d.empty:
                    continue
                n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
                n_zero = int((d == 0).sum())
                bt = binomtest(n_pos, n_pos + n_neg, 0.5) if (n_pos + n_neg) else None
                v = d.to_numpy(dtype=float)
                row = {
                    "metric": metric, "selector": selector, "k": k,
                    "quantity": quantity, "pair": f"{hi_lv}-{lo_lv}",
                    "n_targets": int(len(d)),
                    "n_higher": n_pos, "n_lower": n_neg, "n_exact_tie": n_zero,
                    "mean_diff": float(v.mean()),
                    "sign_test_p": (float(bt.pvalue) if bt is not None else np.nan),
                }
                row.update(widest_ci(v, d.index.tolist(), cl))
                signs.append(row)
    return summary, pd.DataFrame(signs)


# ---------------------------------------------------------------- main

def main() -> int:
    global E04G_DIR                       
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-l3", action="store_true",
                    help="L3 e04g 20/20")
    ap.add_argument("--exclude-gin", action="store_true",
                    help="123 gin --allow-l3")
    ap.add_argument("--gnn-agg", default="seed_median",
                    choices=["seed_median", "fold_mean_only"],
                    help="§9 median fold mean"
                         "fold_mean_only")
    ap.add_argument("--e04g-dir", default=E04G_DIR,
                    help=f"GNN default {E04G_DIR}"
                         f"e04g_hparam_pool_gnn_merged scripts/e04g_merge_topup.py")
    ap.add_argument("--tag", default=None,
                    help="default")
    args = ap.parse_args()
    E04G_DIR = args.e04g_dir
    if E04G_DIR != "e04g_hparam_pool_gnn":
        print(f"[e04ld] ⚠ GNN = {E04G_DIR}", flush=True)

    ok_gnn, missing, gnn_detail = e04g_complete(verbose=args.allow_l3)
    with_l3 = False
    if args.allow_l3:
        if not ok_gnn:
            print("[e04ld] ⛔ L3 e04g GNN"
                  f"{len(missing)}/20 -> {missing}")
            print("[e04ld] §6 7 2 L3 data")
            return 2
        with_l3 = True
    else:
        n_ok = 20 - len(missing)
        print(f"[e04ld] L3 default e04g ****target = {n_ok}/20"
              f"= 72×(8+) ⛔ ⛔ 1080")

    families = dict(TREE_FAMILIES)
    if with_l3:
        gnn = [a for a in GNN_ALGOS if not (args.exclude_gin and a == "gin")]
        families["gnn"] = gnn
    algos = [a for fam in families.values() for a in fam]
    scope = ("trees_only_4algo_2family" if not with_l3
             else f"full_{len(algos)}algo_{len(families)}family"
                  f"{'_no_gin' if args.exclude_gin else '_with_gin'}"
                  f"__{args.gnn_agg}")
    print(f"[e04ld] = {scope} algos = {algos}")

    rows, used, diags = [], [], {}
    for slug in FROZEN20:
        try:
            agg, diag = build_agg(slug, include_gnn=with_l3,
                                  gnn_agg=args.gnn_agg,
                                  exclude_gin=args.exclude_gin)
        except FileNotFoundError as e:
            print(f"[e04ld] ⚠ {slug} {e} skip")
            continue
        
        
        if diag["n_nan_unexplained"]:
            print(f"[e04ld] ⛔ {slug} {diag['n_nan_unexplained']} NaN"
                  f"{diag['unexplained'][:8]}")
            return 1
        pool_models = agg[agg.algo.isin(algos)]
        if pool_models[METRICS].isna().any().any():
            print(f"[e04ld] ⛔ {slug} NaN")
            return 1
        diags[slug] = {k: v for k, v in diag.items() if k != "unexplained"}
        rows += analyse_target(slug, agg, algos=algos,
                               families=families, with_l3=with_l3)
        used.append(slug)
    per_target = pd.DataFrame(rows)
    if per_target.empty:
        print("[e04ld] ⛔ noavailabledata")
        return 1
    print(f"[e04ld] target {len(used)}/20 per_target {len(per_target)}")

    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")
    absent = sorted(set(used) - set(cl.index))
    if absent:
        
        print(f"[e04ld] ⛔ target {CLUSTERS.name} resample {absent}")
        return 1
    summary, signs = summarise(per_target, cl)

    outdir = OUTDIR / (args.tag or scope)
    outdir.mkdir(parents=True, exist_ok=True)
    save_table(per_target, outdir / "per_target_level.csv", float_fmt="%.6f")
    save_table(summary, outdir / "level_summary.csv", float_fmt="%.6f")
    save_table(signs, outdir / "level_sign_tests.csv", float_fmt="%.6f")

    write_run_meta(
        outdir, target="cross_target", stage="e04_level_dissociation",
        protocol_fingerprint=f"selectors={SELECTORS};eval={EVAL_PROTOCOL}",
        extra={
            "declared_in": DECLARED_IN,
            "scope": scope,
            "with_l3": with_l3,
            "levels_computed": sorted(per_target["level"].unique().tolist()),
            "k_main": K_MAIN, "k_secondary": K_SECONDARY,
            "metrics": METRICS,
            "n_targets": len(used), "targets": used,
            "e04g_complete": ok_gnn,
            "e04g_missing": missing,
            "e04g_incomplete_detail": gnn_detail,
            "e04g_expected_rows_per_target": {t: _expected_rows(t)[0] for t in FROZEN20},
            "e04g_gate_note": ("message"
                               "⛔ per_fold.csv"),
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "boot_units": [u for u, _ in BOOT_UNITS],
            "boot_rule": "CI 112",
            "boot_impl": "e20_two_stage._boot_ci e19/e20",
            "source": (["results/<target>/e04_hparam_pool/per_fold.csv"]
                       + ([f"results/<target>/{E04G_DIR}/per_fold.csv"]
                          if with_l3 else [])),
            "source_precision_note": (
                "per_fold.csv save_table %.4f 4"
                "frac_tied 114"
                + ("GNN e04g per_fold.csv to_csv ****"
                   "float_precision='round_trip' 186"
                   "⛔ ⛔"
                   if with_l3 else "")),
            "prereg_status": (
                ("§4 **** 1 3 7 algo"
                 "e04g 20/20 ⛔ 123 gin / gin"
                 "⛔"
                 if with_l3 else
                 "§4 **** 1 3 7 algo"
                 "gnn e04g")),
            "n_threads": N_THREADS,
            "gnn_agg": args.gnn_agg,
            "exclude_gin": bool(args.exclude_gin),
            "families": {k: v for k, v in families.items()},
            "collapse_rule": ("3 pred_std==0 n_pred_nan==0 spearman/kendall"
                              "0 ⛔ dropna"
                              "NaN"),
            "per_target_diagnostics": diags,
        })

    
    pd.set_option("display.width", 240)
    print("\n===== k=2 §2 CI = =====")
    m = summary[(summary.k == K_MAIN) & (summary.metric == "spearman")]
    print(m[["selector", "level", "quantity", "n_targets", "mean", "ci_lo",
             "ci_hi", "ci_width", "boot_unit_widest"]].to_string(index=False))
    print("\n===== k=2, spearman =====")
    s = signs[(signs.k == K_MAIN) & (signs.metric == "spearman")]
    print(s[["selector", "quantity", "pair", "n_higher", "n_lower", "mean_diff",
             "ci_lo", "ci_hi", "boot_unit_widest",
             "sign_test_p"]].to_string(index=False))
    if with_l3:
        print(f"\n[e04ld] §4 **** 3 7 algo = {scope}"
              "\n[e04ld] ⛔ 123 gin / gin ⛔")
    else:
        print("\n[e04ld] ⛔ §4 **** gnn L3")
    print(f"[e04ld] {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
