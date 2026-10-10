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
from scipy.stats import wilcoxon  # noqa: E402

from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import save_table, write_run_meta  # noqa: E402

from e01_regret_analysis import FROZEN20, _metrics, _pick, regret  # noqa: E402
from e14_loto_source import (  # noqa: E402
    CONSERVATIVE_INFLATION, N_BOOT, _prescribe, _selfpick,
)
from e20_two_stage import _boot_ci  

DECLARED_IN = "manuscript protocol declaration"

GNN_DIR = ROOT / "results" / "r2_gnn_v2"
TREE_CSV = ROOT / "results" / "cross_target" / "e01_regret" / "e01_foldmean_all_metrics.csv"
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"
OUT_DIR = ROOT / "results" / "cross_target" / "r2_gnn_regret"


TREES4 = ["lightgbm", "hgb", "rf", "extratrees"]
GNN3 = ["dmpnn", "attentivefp", "gin"]
MIXED7 = TREES4 + GNN3            
GNN2 = ["dmpnn", "attentivefp"]   
MIXED6 = TREES4 + GNN2            





ALL7 = ["lightgbm", "hgb", "rf", "extratrees", "ridge", "svr", "knn"]

METRICS = ["spearman", "kendall", "enrich5", "enrich10"]   
SELECTORS = ["random_holdout", "scaffold_cv"]              
TEMPORAL = "temporal"
PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]


SUBSTITUTE = {"spearman": 0.0, "kendall": 0.0, "enrich5": 1.0, "enrich10": 1.0}


BOOT_SEED = 20260826              



HEADLINE = {"comparison": "G1", "pool": "mixed7", "metric": "spearman"}


LANDED_FOLDMEAN_SPEARMAN = {
    ("attentivefp", "random_holdout"): 0.8049, ("attentivefp", "scaffold_cv"): 0.7136,
    ("attentivefp", "temporal"): 0.4149,
    ("dmpnn", "random_holdout"): 0.8308, ("dmpnn", "scaffold_cv"): 0.7470,
    ("dmpnn", "temporal"): 0.4284,
    ("gin", "random_holdout"): 0.7118, ("gin", "scaffold_cv"): 0.6604,
    ("gin", "temporal"): 0.3805,
}
GATE1_TOL = 5e-5                  
GATE2_TOL = 1e-6
TREE_GATE_TOL = 5e-5              






EXPECT_N_ROWS = 2592              
EXPECT_N_CONSTANT = 68            



def load_gnn_per_fold() -> pd.DataFrame:
    fs = sorted(GNN_DIR.glob("r2_per_fold_*.csv"))
    if not fs:
        raise FileNotFoundError(GNN_DIR / "r2_per_fold_*.csv")
    pf = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return pf


def load_gnn_predictions() -> pd.DataFrame:
    fs = sorted(GNN_DIR.glob("r2_predictions_*.csv"))
    if not fs:
        raise FileNotFoundError(GNN_DIR / "r2_predictions_*.csv")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _agg_seed_then_fold(pf: pd.DataFrame, col: str) -> pd.DataFrame:
    """Documentation."""
    per_fold = (pf.groupby(["model", "target", "protocol", "fold"])[col]
                .median().reset_index())
    return (per_fold.groupby(["model", "target", "protocol"])[col]
            .mean().reset_index())



def gate1_landed_foldmean(pf: pd.DataFrame) -> bool:
    """Documentation."""
    fm = _agg_seed_then_fold(pf, "spearman")
    piv = fm.pivot_table(index="model", columns="protocol", values="spearman")
    ok = True
    for (model, protocol), ref in sorted(LANDED_FOLDMEAN_SPEARMAN.items()):
        got = float(piv.loc[model, protocol])
        if abs(got - ref) > GATE1_TOL:
            print(f"[gate1] {model}/{protocol}: {got:.6f} ≠ note_8/25 {ref:.4f}",
                  file=sys.stderr)
            ok = False
    return ok


def gate2_recompute_matches(rec: pd.DataFrame, pf: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    key = ["target", "protocol", "model", "fold", "seed"]
    m = rec.merge(pf[key + ["spearman", "enrich5", "enrich10", "pred_std"]],
                  on=key, suffixes=("_rec", "_pf"))
    if len(m) != len(pf):
        return False, [f"{len(m)} ≠ per_fold {len(pf)}"]
    ok, msg = True, []
    
    sub = m[m["spearman_pf"].notna() & m["spearman_rec"].notna()]
    d = (sub["spearman_rec"] - sub["spearman_pf"]).abs().max()
    msg.append(f"spearman : {len(sub)} {d:.3e}")
    if not np.isfinite(d) or d > GATE2_TOL:
        ok = False
        msg[-1] += "<<"
    
    for col in ("enrich5", "enrich10"):
        s = m[m[f"{col}_pf"].notna() & m[f"{col}_rec"].notna() & (m["pred_std"] != 0)]
        diff = (s[f"{col}_rec"] - s[f"{col}_pf"]).abs() > GATE2_TOL
        bad = int((diff & ~s["boundary_tie_" + col]).sum())
        msg.append(f"{col}: {len(s)} {int(diff.sum())}"
                   f"{(s[f'{col}_rec'] - s[f'{col}_pf']).abs().max():.4f}"
                   f"{bad}")
        if bad:
            ok = False
            msg[-1] += "<< ⇒ inconsistent"
    
    const = m[m["pred_std"] == 0]
    msg.append(f"prediction (pred_std==0): {len(const)}")
    if len(const):
        nan_rec = int(const["spearman_rec"].isna().sum())
        e5 = const["enrich5_pf"]
        msg.append(f"spearman NaN: {nan_rec}/{len(const)}")
        msg.append(f"per_fold enrich5 {e5.min():.4f}–{e5.max():.4f}"
                   f"{e5.nunique()} ⇒ artifact §3.1")
        if nan_rec != len(const):
            ok = False
            msg.append("<< NaN")
    return ok, msg


def gate3_tree_matches(rec_tree: pd.DataFrame, landed: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    on = ["target", "protocol", "model"]
    m = rec_tree.merge(landed, on=on, suffixes=("_rec", "_e01"))
    if len(m) != len(landed):
        return False, [f"{len(m)} ≠ e01 {len(landed)}"]
    ok, msg = True, []
    for col in ("spearman", "kendall"):
        d = (m[f"{col}_rec"] - m[f"{col}_e01"]).abs().max()
        msg.append(f"{col} : {len(m)} {d:.3e}")
        if not np.isfinite(d) or d > TREE_GATE_TOL:
            ok = False
            msg[-1] += "<< e01"
    for col in ("enrich5", "enrich10"):
        dd = (m[f"{col}_rec"] - m[f"{col}_e01"]).abs()
        msg.append(f"{col} : {int((dd > 1e-9).sum())}/{len(m)}"
                   f"(target,protocol,model) mean mol_id {dd.max():.4f}")
    
    
    spurious = 0
    for (_t, _p), g in m.groupby(["target", "protocol"]):
        for metric in METRICS:
            for pool in (TREES4,):
                s_rec = g.set_index("model")[f"{metric}_rec"].reindex(pool).dropna()
                s_e01 = g.set_index("model")[f"{metric}_e01"].reindex(pool).dropna()
                if len(s_rec) < 2:
                    continue
                ties_e01 = int(np.isclose(s_e01, s_e01.max(), atol=1e-12).sum())
                ties_rec = int(np.isclose(s_rec, s_rec.max(), atol=1e-12).sum())
                if ties_e01 > ties_rec:
                    spurious += 1
    msg.append(f"★ %.4f {spurious} (target,protocol,metrics)"
               f"⇒ model"
               f"enrich consistent")
    return ok, msg


def _per_fold_metrics(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for k, g in df.groupby(keys, sort=False):
        g = g.sort_values("mol_id", kind="mergesort")
        yt, yp = g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float)
        n = len(g)
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n"] = n
        r.update(_metrics(yt, yp))
        for pct in (5, 10):
            kk = max(1, int(round(n * pct / 100.0)))
            sp, st = np.sort(yp)[::-1], np.sort(yt)[::-1]
            tie_p = bool(kk < n and sp[kk - 1] == sp[kk])
            tie_t = bool(kk < n and st[kk - 1] == st[kk])
            r[f"boundary_tie_enrich{pct}"] = tie_p or tie_t
            r[f"boundary_tie_true_enrich{pct}"] = tie_t
            r[f"boundary_tie_pred_enrich{pct}"] = tie_p
        rows.append(r)
    return pd.DataFrame(rows)


def build_tree_foldmean_recomputed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Documentation."""
    per_fold, fm = [], []
    for slug in FROZEN20:
        p = ROOT / "results" / slug / "p03_protocol_matrix" / "predictions.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        pfd = _per_fold_metrics(pd.read_csv(p), ["protocol", "model", "fold"])
        pfd.insert(0, "target", slug)
        per_fold.append(pfd)
        agg = (pfd.groupby(["protocol", "model"])[METRICS].mean().reset_index())
        agg.insert(0, "target", slug)
        agg["n_folds"] = (pfd.groupby(["protocol", "model"])["spearman"]
                          .count().reset_index(drop=True))
        fm.append(agg)
    return pd.concat(per_fold, ignore_index=True), pd.concat(fm, ignore_index=True)


def load_tree_foldmean_landed() -> pd.DataFrame:
    """Documentation."""
    t = pd.read_csv(TREE_CSV)
    return t[["target", "protocol", "model"] + METRICS].copy()



def build_gnn_foldmean(rec: pd.DataFrame, pred_std: pd.DataFrame, *,
                       variant: str) -> pd.DataFrame:
    """Documentation."""
    key = ["target", "protocol", "model", "fold", "seed"]
    d = rec.merge(pred_std, on=key, how="left")
    if variant == "V2":
        d = d[d["seed"] == 0].copy()
    if variant != "V3":
        const = d["pred_std"] == 0
        for metric, val in SUBSTITUTE.items():
            d.loc[const, metric] = val
    out = []
    for metric in METRICS:
        fm = _agg_seed_then_fold(d, metric).rename(columns={metric: "value"})
        fm["metric"] = metric
        out.append(fm)
    long = pd.concat(out, ignore_index=True)
    wide = long.pivot_table(index=["target", "protocol", "model"],
                            columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide


def load_tree_foldmean() -> pd.DataFrame:
    """Documentation."""
    t = pd.read_csv(TREE_CSV)
    return t[["target", "protocol", "model"] + METRICS].copy()



def regret_table(tab: pd.DataFrame, pools: dict[str, list[str]]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for slug, agg in tab.groupby("target"):
        for pname, pool in pools.items():
            for metric in METRICS:
                for selector in SELECTORS:
                    r = regret(agg, selector=selector, pool=pool, metric=metric)
                    if r:
                        rows.append({"target": slug, "pool": pname, **r})
    return pd.DataFrame(rows)


def loto_table(tab: pd.DataFrame, pools: dict[str, list[str]]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    targets = sorted(tab["target"].unique())
    for pname, pool in pools.items():
        for metric in METRICS:
            for held in targets:
                tmp = (tab[(tab["protocol"] == TEMPORAL) & (tab["target"] == held)]
                       .set_index("model")[metric].reindex(pool).dropna())
                if len(tmp) < 2:
                    continue
                tbest = str(tmp.idxmax())
                row = {"pool": pname, "metric": metric, "target": held,
                       "pool_size": len(tmp), "temporal_best_model": tbest,
                       "temporal_best": float(tmp.max()),
                       "random_pick_regret": float(np.mean(tmp.max() - tmp))}
                for source in PROTOCOLS:
                    m_star, n_ties = _prescribe(tab, pool=pool, metric=metric,
                                                source=source, held=held)
                    if m_star and m_star in tmp.index:
                        row[f"loto_{source}_model"] = m_star
                        row[f"loto_{source}_regret"] = float(tmp.max() - tmp[m_star])
                        row[f"loto_{source}_n_ties"] = n_ties
                sp, sp_ties = _selfpick(tab, pool=pool, metric=metric, target=held)
                if sp and sp in tmp.index:
                    row["cv_selfpick_model"] = sp
                    row["cv_selfpick_regret"] = float(tmp.max() - tmp[sp])
                    row["cv_selfpick_n_ties"] = sp_ties
                rows.append(row)
    return pd.DataFrame(rows)



def _paired(reg: pd.DataFrame, lo: pd.DataFrame) -> list[dict]:
    """Documentation."""
    out = []

    def add(cid, question, arm_a, arm_b, pool_a, pool_b, metric, selector, df_a, df_b):
        m = df_a.merge(df_b, on="target", suffixes=("_a", "_b"))
        if len(m) < 2:
            return
        out.append({
            "comparison": cid, "question": question,
            "arm_a": arm_a, "arm_b": arm_b, "pool_a": pool_a, "pool_b": pool_b,
            "metric": metric, "selector": selector,
            "targets": m["target"].tolist(),
            "delta": (m["v_a"] - m["v_b"]).to_numpy(dtype=float),
            "direction_note": "= arm_a",
        })

    def rg(pool, metric, selector):
        s = reg[(reg["pool"] == pool) & (reg["metric"] == metric)
                & (reg["selector"] == selector)]
        return s[["target", "regret"]].rename(columns={"regret": "v"})

    def rp(pool, metric, selector):
        s = reg[(reg["pool"] == pool) & (reg["metric"] == metric)
                & (reg["selector"] == selector)]
        return s[["target", "random_pick_regret"]].rename(
            columns={"random_pick_regret": "v"})

    def lt(pool, metric, col):
        s = lo[(lo["pool"] == pool) & (lo["metric"] == metric)]
        if col not in s.columns:
            return pd.DataFrame(columns=["target", "v"])
        return s[["target", col]].rename(columns={col: "v"}).dropna()

    pools_present = sorted(reg["pool"].unique())
    mixed = "mixed7" if "mixed7" in pools_present else "mixed6"
    gnn = "gnn3" if "gnn3" in pools_present else "gnn2"

    for metric in METRICS:
        for selector in SELECTORS:
            
            add("G1", "message",
                f"{mixed}_regret", "trees4_regret", mixed, "trees4", metric, selector,
                rg(mixed, metric, selector), rg("trees4", metric, selector))
            # G2 gnn3 vs trees4
            add("G2", "GNN",
                f"{gnn}_regret", "trees4_regret", gnn, "trees4", metric, selector,
                rg(gnn, metric, selector), rg("trees4", metric, selector))
            
            for pool in pools_present:
                add("G3", "message",
                    f"{pool}_regret", f"{pool}_random_pick", pool, pool, metric, selector,
                    rg(pool, metric, selector), rp(pool, metric, selector))
        
        for pool in pools_present:
            add("G4", "scaffoldrandom",
                f"{pool}_scaffold_cv", f"{pool}_random_holdout", pool, pool,
                metric, "both", rg(pool, metric, "scaffold_cv"),
                rg(pool, metric, "random_holdout"))
        
        for pool in pools_present:
            add("G5", "model vs CV e14 H1",
                f"{pool}_loto_random_holdout", f"{pool}_cv_selfpick", pool, pool,
                metric, "loto", lt(pool, metric, "loto_random_holdout_regret"),
                lt(pool, metric, "cv_selfpick_regret"))
            add("G6", "model vs e14 H2",
                f"{pool}_loto_random_holdout", f"{pool}_random_pick", pool, pool,
                metric, "loto", lt(pool, metric, "loto_random_holdout_regret"),
                lt(pool, metric, "random_pick_regret"))
    return out


def _t_needed(delta: np.ndarray) -> float:
    """Documentation."""
    n = len(delta)
    if n < 2:
        return np.nan
    sd, mean = float(np.std(delta, ddof=1)), float(np.mean(delta))
    if not np.isfinite(sd) or not np.isfinite(mean) or abs(mean) <= 1e-12:
        return np.nan
    return float(np.ceil((1.96 * sd / abs(mean)) ** 2))


def analyse(comparisons: list[dict], clusters: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    cl = clusters.set_index("slug")
    rows = []
    for c in comparisons:
        d = np.asarray(c["delta"], dtype=float)
        tg = c["targets"]
        keep = np.isfinite(d)
        d, tg = d[keep], [t for t, k in zip(tg, keep) if k]
        if len(d) < 2:
            continue
        base = {k: c[k] for k in ("comparison", "question", "arm_a", "arm_b",
                                  "pool_a", "pool_b", "metric", "selector",
                                  "direction_note")}
        n_pos = int((d > 1e-12).sum())
        n_neg = int((d < -1e-12).sum())
        base.update({"n_targets": len(d), "delta_mean": float(d.mean()),
                     "delta_sd": float(d.std(ddof=1)),
                     "n_pos": n_pos, "n_neg": n_neg,
                     "n_tie": len(d) - n_pos - n_neg,
                     "t_needed": _t_needed(d),
                     "sd_rel_se": float(1.0 / np.sqrt(2 * (len(d) - 1)))})
        
        try:
            base["wilcoxon_p"] = float(
                wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            base["wilcoxon_p"] = np.nan
        
        cis = {}
        for unit in ("target", "cluster_fine", "cluster_coarse"):
            units = None if unit == "target" else\
                cl.reindex(tg)[unit].to_numpy()
            if units is not None and pd.isna(units).any():
                continue
            ci = _boot_ci(d, units, np.random.default_rng(BOOT_SEED))
            cis[unit] = ci
            base[f"ci_lo_raw_{unit}"] = ci["ci_lo"]
            base[f"ci_hi_raw_{unit}"] = ci["ci_hi"]
            base[f"ci_lo_cons_{unit}"] = ci["ci_lo_cons"]
            base[f"ci_hi_cons_{unit}"] = ci["ci_hi_cons"]
            base[f"n_units_{unit}"] = ci["n_units"]
        if not cis:
            continue
        
        widest = max(cis, key=lambda u: cis[u]["ci_hi_cons"] - cis[u]["ci_lo_cons"])
        w = cis[widest]
        base.update({
            "widest_unit": widest,
            "ci_lo_cons": w["ci_lo_cons"], "ci_hi_cons": w["ci_hi_cons"],
            "ci_lo_raw": w["ci_lo"], "ci_hi_raw": w["ci_hi"],
            "ci_width_cons": w["ci_hi_cons"] - w["ci_lo_cons"],
            "n_units_widest": w["n_units"],
            "excludes_zero_cons": bool(w["ci_lo_cons"] > 0 or w["ci_hi_cons"] < 0),
            
            
            "excludes_zero_raw": bool(w["ci_lo"] > 0 or w["ci_hi"] < 0),
            "verdict_hinges_on_inflation": bool(
                (w["ci_lo"] > 0 or w["ci_hi"] < 0)
                != (w["ci_lo_cons"] > 0 or w["ci_hi_cons"] < 0)),
            "is_headline": (c["comparison"] == HEADLINE["comparison"]
                            and c["pool_a"] == HEADLINE["pool"]
                            and c["metric"] == HEADLINE["metric"]),
        })
        rows.append(base)
    return pd.DataFrame(rows)


# ===================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description="gnn3/mixed7 note_8/26")
    ap.add_argument("--variants", nargs="*",
                    default=["V0", "V1", "V2", "V3"],
                    help="V0 / V1 -gin / V2 / V3")
    args = ap.parse_args()

    assert N_BOOT == 10000, f"§8.1 N_BOOT=10000 {N_BOOT}"
    assert abs(CONSERVATIVE_INFLATION - 1.222) < 1e-12, "§8.1 1.222"

    prot = load_protocol()
    print(f"[r2gr] = {DECLARED_IN}")
    print(f"[r2gr] N_BOOT={N_BOOT} BOOT_SEED={BOOT_SEED} "
          f"CONSERVATIVE_INFLATION={CONSERVATIVE_INFLATION}")

    
    pf = load_gnn_per_fold()
    print(f"[r2gr] GNN per_fold {pf.shape} {EXPECT_N_ROWS}")
    if len(pf) != EXPECT_N_ROWS:
        print(f"[r2gr] !! per_fold note_8/25", file=sys.stderr)
        return 1
    n_const = int((pf["pred_std"] == 0).sum())
    print(f"[r2gr] pred_std==0 · {n_const} note_8/25 {EXPECT_N_CONSTANT}")
    if n_const != EXPECT_N_CONSTANT:
        print(f"[r2gr] !!", file=sys.stderr)
        return 1

    print("[r2gr] 1 note_8/25 9 spearman")
    if not gate1_landed_foldmean(pf):
        print("[r2gr] 1 failed", file=sys.stderr)
        return 1
    print("[r2gr] 1 9/9")

    print("[r2gr] moleculespredictionmetrics §11.1 kendall e01 _metrics")
    preds = load_gnn_predictions()
    print(f"[r2gr] prediction {preds.shape}")
    rec = _per_fold_metrics(preds, ["target", "protocol", "model", "fold", "seed"])
    print(f"[r2gr] {rec.shape}")

    print("[r2gr] 2 vs per_fold")
    ok2, msg2 = gate2_recompute_matches(rec, pf)
    for m in msg2:
        print(m)
    if not ok2:
        print("[r2gr] 2 failed", file=sys.stderr)
        return 1
    print("[r2gr] 2")

    print("[r2gr] GNN + mol_id")
    tree_pf, tree = build_tree_foldmean_recomputed()
    landed = load_tree_foldmean_landed()
    print(f"[r2gr] {tree_pf.shape} {tree.shape}")
    print("[r2gr] 3 vs e01")
    ok3, msg3 = gate3_tree_matches(tree, landed)
    for m in msg3:
        print(m)
    if not ok3:
        print("[r2gr] 3 failed", file=sys.stderr)
        return 1
    print("[r2gr] 3")

    
    n_tree_const = int((tree_pf["spearman"].isna()).sum())
    print(f"[r2gr] NaN = prediction n<3 {n_tree_const}/{len(tree_pf)}"
          f"0 GNN")

    
    
    t_gnn, t_tree = set(rec["target"]), set(tree["target"])
    if t_gnn != set(FROZEN20) or t_tree != set(FROZEN20):
        print(f"[r2gr] !! target GNN {sorted(set(FROZEN20) - t_gnn)}"
              f"{sorted(set(FROZEN20) - t_tree)}", file=sys.stderr)
        return 1
    tree = tree[["target", "protocol", "model"] + METRICS].copy()

    clusters = pd.read_csv(CLUSTERS, comment="#")
    clusters = clusters[clusters["slug"].isin(FROZEN20)]
    print(f"[r2gr] {len(clusters)} {clusters['cluster_fine'].nunique()} fine /"
          f"{clusters['cluster_coarse'].nunique()} coarse")

    pred_std = pf[["target", "protocol", "model", "fold", "seed", "pred_std"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats, all_reg, all_lo = [], [], []

    for variant in args.variants:
        gnn_pools = {"gnn2" if variant == "V1" else "gnn3":
                     GNN2 if variant == "V1" else GNN3,
                     "mixed6" if variant == "V1" else "mixed7":
                     MIXED6 if variant == "V1" else MIXED7}
        pools = {"trees4": TREES4, **gnn_pools}
        gv = "V0" if variant == "V1" else variant   
        gnn_fm = build_gnn_foldmean(rec, pred_std, variant=gv)
        tab = pd.concat([tree, gnn_fm], ignore_index=True)

        reg = regret_table(tab, pools)
        lo = loto_table(tab, pools)
        stats = analyse(_paired(reg, lo), clusters)
        for df, lst in ((reg, all_reg), (lo, all_lo), (stats, all_stats)):
            df.insert(0, "variant", variant)
            lst.append(df)
        print(f"[r2gr] {variant}: {list(pools)} regret {len(reg)}"
              f"loto {len(lo)} {len(stats)}")

    reg = pd.concat(all_reg, ignore_index=True)
    lo = pd.concat(all_lo, ignore_index=True)
    stats = pd.concat(all_stats, ignore_index=True)

    save_table(reg, OUT_DIR / "r2gr_regret_by_target.csv", float_fmt="%.6g")
    save_table(lo, OUT_DIR / "r2gr_loto_by_target.csv", float_fmt="%.6g")
    save_table(stats, OUT_DIR / "r2gr_comparisons.csv", float_fmt="%.6g")

    
    
    sens = []
    for arm, tab_pf in (("gnn", rec), ("trees", tree_pf)):
        for pct in (5, 10):
            col = f"enrich{pct}"
            n = len(tab_pf)
            sens.append({
                "arm": arm, "metric": col, "n_folds": n,
                "n_boundary_tie": int(tab_pf[f"boundary_tie_{col}"].sum()),
                "n_boundary_tie_true_only": int((tab_pf[f"boundary_tie_true_{col}"]
                                                 & ~tab_pf[f"boundary_tie_pred_{col}"]).sum()),
                "n_boundary_tie_pred": int(tab_pf[f"boundary_tie_pred_{col}"].sum()),
                "frac_boundary_tie": float(tab_pf[f"boundary_tie_{col}"].mean()),
                "note": "top-k ⇒ enrich"
                        "mol_id",
            })
    sens = pd.DataFrame(sens)
    save_table(sens, OUT_DIR / "r2gr_enrich_tie_sensitivity.csv", float_fmt="%.6g")
    print("\n[r2gr] enrich")
    print(sens[["arm", "metric", "n_folds", "n_boundary_tie",
                "n_boundary_tie_true_only", "n_boundary_tie_pred",
                "frac_boundary_tie"]].to_string(index=False))

    
    gnn_fm_v0 = build_gnn_foldmean(rec, pred_std, variant="V0")
    tab_v0 = pd.concat([tree, gnn_fm_v0], ignore_index=True)
    ph_pools = {"all7": ALL7, "mixed7": MIXED7, "trees4": TREES4}
    ph_reg = regret_table(tab_v0, ph_pools)
    ph = []
    for metric in METRICS:
        for selector in SELECTORS:
            def _v(pool):
                s = ph_reg[(ph_reg["pool"] == pool) & (ph_reg["metric"] == metric)
                           & (ph_reg["selector"] == selector)]
                return s[["target", "regret"]].rename(columns={"regret": "v"})
            for a, b in (("mixed7", "all7"), ("all7", "trees4")):
                m = _v(a).merge(_v(b), on="target", suffixes=("_a", "_b"))
                d = (m["v_a"] - m["v_b"]).to_numpy(float)
                ci = _boot_ci(d, None, np.random.default_rng(BOOT_SEED))
                ph.append({"comparison": f"POSTHOC_{a}_vs_{b}", "metric": metric,
                           "selector": selector, "n_targets": len(d),
                           "size_matched": a == "mixed7" and b == "all7",
                           "delta_mean": float(d.mean()),
                           "n_pos": int((d > 1e-12).sum()),
                           "n_neg": int((d < -1e-12).sum()),
                           "n_tie": int((np.abs(d) <= 1e-12).sum()),
                           "ci_lo_cons": ci["ci_lo_cons"], "ci_hi_cons": ci["ci_hi_cons"],
                           "excludes_zero_cons": bool(ci["ci_lo_cons"] > 0
                                                      or ci["ci_hi_cons"] < 0),
                           "direction_note": f"= {a}"})
    ph = pd.DataFrame(ph)
    save_table(ph, OUT_DIR / "r2gr_POSTHOC_size_matched.csv", float_fmt="%.6g")
    print("\n[r2gr] ★ §7 main table")
    print("G1 7 vs 4 e01")
    print("mixed7 vs all7 7 7")
    print(ph[ph["metric"] == "spearman"][
        ["comparison", "selector", "delta_mean", "n_pos", "n_neg", "n_tie",
         "ci_lo_cons", "ci_hi_cons", "excludes_zero_cons"]].to_string(index=False))
    print("spearman / random_holdout")
    _v0 = ph_reg[(ph_reg["metric"] == "spearman")
                 & (ph_reg["selector"] == "random_holdout")]
    print(_v0.groupby("pool").agg(pool_size=("pool_size", "max"),
                                  mean_regret=("regret", "mean"),
                                  mean_noise_floor=("random_pick_regret", "mean")
                                  ).to_string())

    
    print("\n" + "=" * 78)
    print(f"★ §7 {HEADLINE['comparison']} ×"
          f"{HEADLINE['pool']} × {HEADLINE['metric']} V0")
    print("=" * 78)
    h = stats[(stats["variant"] == "V0") & stats["is_headline"]]
    cols = ["selector", "n_targets", "delta_mean", "delta_sd", "n_pos", "n_neg", "n_tie",
            "wilcoxon_p", "widest_unit", "ci_lo_raw", "ci_hi_raw", "excludes_zero_raw",
            "ci_lo_cons", "ci_hi_cons", "excludes_zero_cons",
            "verdict_hinges_on_inflation", "t_needed", "sd_rel_se"]
    print(h[cols].to_string(index=False))
    if bool(h["verdict_hinges_on_inflation"].any()):
        print("\n⚠️ ⇒ 1.222"
              "\n §8.1 115")
    if len(h) > 1 and h["excludes_zero_cons"].nunique() > 1:
        print("\n⚠️⚠️ §7 comparison × pool × metric **no selector**"
              "\n selector 0 note_8/27 §4"
              "\n **** §9 selector"
              "\n selector")
    print(f"\n {h['direction_note'].iloc[0] if len(h) else 'n/a'}")
    print("resample 93+112")
    for _, r in h.iterrows():
        for unit in ("target", "cluster_fine", "cluster_coarse"):
            lo_c, hi_c = r.get(f"ci_lo_cons_{unit}"), r.get(f"ci_hi_cons_{unit}")
            if pd.notna(lo_c):
                print(f"  {r['selector']:16s} {unit:15s} "
                      f"[{lo_c:+.4f}, {hi_c:+.4f}] {hi_c - lo_c:.4f}"
                      f"n_units={int(r[f'n_units_{unit}'])}")

    print(f"\n[r2gr] {len(stats)} {OUT_DIR}")
    print("[r2gr] ⚠️ §7"
          "p")

    write_run_meta(OUT_DIR, target="__cross_target__", stage="r2_gnn_regret",
                   protocol_fingerprint=prot.fingerprint,
                   extra={
                       "declared_in": DECLARED_IN,
                       "scope": "hparam_grid_r2_gnn.yaml §0",
                       "no_retrain": True,
                       "inputs": {"gnn_per_fold": str(GNN_DIR),
                                  "tree_foldmean": str(TREE_CSV),
                                  "clusters": str(CLUSTERS)},
                       "gates": {"gate1": "note_8/25 9 spearman 9/9",
                                 "gate2": f"GNN molecules vs per_fold"
                                          f"spearman tol={GATE2_TOL}"
                                          f"enrich",
                                 "gate3": f"vs e01"
                                          f"spearman/kendall tol={TREE_GATE_TOL}"},
                       "both_arms_same_code_path": "_per_fold_metrics + mol_id"
                                                   "enrich",
                       "enrich_row_order_sensitivity": "enrichment_factor top-k y_true"
                                                       "y_pred GNN"
                                                       "288/2524 enrich5"
                                                       "281 y_true ⇒ model"
                                                       "r2gr_enrich_tie_sensitivity.csv",
                       "pools": {"trees4": TREES4, "gnn3": GNN3, "mixed7": MIXED7,
                                 "gnn2_V1": GNN2, "mixed6_V1": MIXED6},
                       "tie_break": "atol=1e-12 e01 _pick"
                                    "mixed7",
                       "substitution": SUBSTITUTE,
                       "n_constant_folds_gnn": n_const,
                       "n_nan_folds_trees": n_tree_const,
                       "seed_aggregation": "median hparam_grid_r2_gnn.yaml §3",
                       "boot": {"N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED,
                                "CONSERVATIVE_INFLATION": CONSERVATIVE_INFLATION,
                                "half_width": "(hi-lo)/2 × 1.222 e14/e20 e18",
                                "units": "target / cluster_fine / cluster_coarse"},
                       "wilcoxon_zero_method": "zsplit",
                       "headline": HEADLINE,
                       "multiplicity": "message",
                       "variants": args.variants,
                       "not_done": ["molecules measurement bootstrap §8.2/§11.3"
                                    "target"
                                    "target §8.2 generalization"
                                    "write default",
                                    "pooled §11.2"],
                   })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
