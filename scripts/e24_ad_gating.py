#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from pparbench.threads import apply_thread_limits  # noqa: E402

N_THREADS = apply_thread_limits()  

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, rankdata  # noqa: E402

from pparbench.ad import bucket_metrics, nearest_neighbour_similarity  # noqa: E402
from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import (  # noqa: E402
    features_path, save_table, target_results_dir, write_run_meta,
)
from p05_ad_analysis import load_ad_fingerprints  # noqa: E402
from e01_regret_analysis import FROZEN20  # noqa: E402
from e14_loto_source import CONSERVATIVE_INFLATION, N_BOOT, POOL_TREES  # noqa: E402
from e15_winner_shift import assert_scope  # noqa: E402
from e20_two_stage import _boot_ci  # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
MANIFEST = ROOT / "configs" / "manifests" / "targets_manifest.csv"
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"
OUTDIR = ROOT / "results" / "cross_target" / "e24_ad_gating"

PROTOCOL = "temporal"                 
TAUS = [0.4, 0.6]                     
HEADLINE = ("trees4", "spearman", 0.4)  
MIN_KEEP = 50                          
B_RAND = 1000                          
RAND_SEED = 20260826                   
BOOT_SEED = 20260826                   
YSD_TOL = 0.05                         
MATCH_MIN_ACCEPT = 200                 
MATCH_MAX_ATTEMPT = 20000              
CHUNK = 500                            
SELF_CHECK_TOL = 5e-4                  



def _rho_rows(ymat: np.ndarray, pmat: np.ndarray) -> np.ndarray:
    """Documentation."""
    ry = rankdata(ymat, axis=1).astype(float)
    rp = rankdata(pmat, axis=1).astype(float)
    ry -= ry.mean(axis=1, keepdims=True)
    rp -= rp.mean(axis=1, keepdims=True)
    den = np.sqrt((ry ** 2).sum(axis=1) * (rp ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ry * rp).sum(axis=1) / den


def _rho(y: np.ndarray, p: np.ndarray) -> float:
    return float(_rho_rows(y[None, :], p[None, :])[0])


def _tau_b(y: np.ndarray, p: np.ndarray) -> float:
    return float(kendalltau(y, p).statistic)


def _n_tied_pairs(v: np.ndarray) -> int:
    """Documentation."""
    _, c = np.unique(v, return_counts=True)
    return int((c * (c - 1) // 2).sum())


def _draw_subsets(rng, n_total: int, n_keep: int, b: int) -> np.ndarray:
    """Documentation."""
    return rng.random((b, n_total)).argsort(axis=1)[:, :n_keep]


def _assert_taus(buckets: list) -> None:
    """Documentation."""
    edges = [float(b[1]) for b in buckets[:-1]]
    if [float(t) for t in TAUS] != edges:
        raise SystemExit(
            f"⛔ τ protocol.yaml inconsistent TAUS={TAUS} vs ={edges}"
            "\n TAUS protocol.yaml"
            "message")



def run_target(slug: str, prot, buckets: list, min_bucket_n: int,
               rng) -> tuple[list, list, list]:
    """Documentation."""
    pred_path = target_results_dir(slug, "p03_protocol_matrix") / "predictions.csv"
    pred = pd.read_csv(pred_path)
    pred = pred[pred["protocol"] == PROTOCOL].copy()
    if pred.empty:
        raise SystemExit(f"⛔ {slug}: predictions.csv no protocol={PROTOCOL}")

    p05_buckets = pd.read_csv(target_results_dir(slug, "p05_ad") / "buckets.csv")
    p05_idx = p05_buckets.set_index(["fold", "model", "bucket"])

    fps = load_ad_fingerprints(slug)
    year = np.load(features_path(slug, "ligand2057"), allow_pickle=True)["year"]

    cells, dropped, problems = [], [], []
    for fold, g_fold in pred.groupby("fold", sort=False):
        if not str(fold).startswith("T"):
            raise SystemExit(f"⛔ {slug}: temporal fold T{{}} {fold}")
        T = int(str(fold)[1:])
        test_idx = np.unique(g_fold["test_idx"].to_numpy())
        train_idx = np.flatnonzero(~np.isnan(year) & (year < T))
        sims_test = nearest_neighbour_similarity(fps, train_idx, test_idx)
        sim_by_idx = dict(zip(test_idx.tolist(), sims_test.tolist()))

        for model, g in g_fold.groupby("model", sort=False):
            s = g["test_idx"].map(sim_by_idx).to_numpy(dtype=float)
            yt = g["y_true"].to_numpy(dtype=float)
            yp = g["y_pred"].to_numpy(dtype=float)
            n_total = len(yt)

            
            
            bm = bucket_metrics(yt, yp, s, buckets, min_n=min_bucket_n)
            for r in bm.itertuples():
                key = (fold, model, r.bucket)
                if key not in p05_idx.index:
                    problems.append(f"{slug} {key} p05 buckets.csv")
                    continue
                ref = p05_idx.loc[key]
                if int(ref["n"]) != int(r.n):
                    problems.append(
                        f"{slug} {key} samplesinconsistent {r.n} vs p05 {int(ref['n'])}")
                a, b_ = r.spearman, float(ref["spearman"])
                if np.isfinite(a) and np.isfinite(b_) and abs(a - b_) > SELF_CHECK_TOL:
                    problems.append(
                        f"{slug} {key} spearman p05 {a:.6f} vs {b_:.4f}")

            y_sd_full = float(yt.std())
            rho_full = _rho(yt, yp)
            tau_full = _tau_b(yt, yp)
            n_nan_sim = int(np.isnan(s).sum())

            for tau in TAUS:
                keep = np.nan_to_num(s, nan=-1.0) >= tau
                n_keep = int(keep.sum())
                if n_keep < MIN_KEEP:
                    dropped.append({
                        "target": slug, "fold": fold, "model": model, "tau": tau,
                        "n_test": n_total, "n_keep": n_keep,
                        "reason": f"< {MIN_KEEP} §3.5"})
                    continue

                ytk, ypk = yt[keep], yp[keep]
                rho_gate, tau_gate = _rho(ytk, ypk), _tau_b(ytk, ypk)
                y_sd_keep = float(ytk.std())

                
                idx = _draw_subsets(rng, n_total, n_keep, B_RAND)
                yr, pr = yt[idx], yp[idx]
                rho_rand = _rho_rows(yr, pr)
                tau_rand = np.array([_tau_b(yr[i], pr[i]) for i in range(B_RAND)])
                ysd_rand = yr.std(axis=1)

                
                tol_abs = YSD_TOL * y_sd_full
                acc = list(np.flatnonzero(np.abs(ysd_rand - y_sd_keep) <= tol_abs))
                acc_rho = list(rho_rand[acc])
                attempts = B_RAND
                while len(acc_rho) < MATCH_MIN_ACCEPT and attempts < MATCH_MAX_ATTEMPT:
                    j = _draw_subsets(rng, n_total, n_keep, CHUNK)
                    yj, pj = yt[j], yp[j]
                    m = np.abs(yj.std(axis=1) - y_sd_keep) <= tol_abs
                    if m.any():
                        acc_rho.extend(_rho_rows(yj[m], pj[m]).tolist())
                    attempts += CHUNK
                matched_ok = len(acc_rho) >= MATCH_MIN_ACCEPT

                cells.append({
                    "target": slug, "fold": fold, "model": model, "tau": tau,
                    "n_test": n_total, "n_keep": n_keep,
                    "coverage": n_keep / n_total,
                    "n_nan_sim": n_nan_sim,
                    "mean_sim_keep": float(np.nanmean(s[keep])),
                    "spearman_full": rho_full,
                    "spearman_gate": rho_gate,
                    "spearman_rand_mean": float(rho_rand.mean()),
                    "spearman_rand_p2p5": float(np.percentile(rho_rand, 2.5)),
                    "spearman_rand_p97p5": float(np.percentile(rho_rand, 97.5)),
                    "delta_spearman": rho_gate - float(rho_rand.mean()),
                    "kendall_full": tau_full,
                    "kendall_gate": tau_gate,
                    "kendall_rand_mean": float(tau_rand.mean()),
                    "delta_kendall": tau_gate - float(tau_rand.mean()),
                    "y_sd_full": y_sd_full,
                    "y_sd_keep": y_sd_keep,
                    "y_sd_rand_mean": float(ysd_rand.mean()),
                    "delta_spearman_ysd_matched": (
                        rho_gate - float(np.mean(acc_rho)) if matched_ok else np.nan),
                    "n_matched_draws": len(acc_rho),
                    "matched_evaluable": matched_ok,
                    "n_tied_pairs_pred_keep": _n_tied_pairs(ypk),
                })
    return cells, dropped, problems



def aggregate(cells: pd.DataFrame, clusters: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Documentation."""
    qty = ["delta_spearman", "delta_kendall", "delta_spearman_ysd_matched",
           "coverage", "y_sd_keep", "y_sd_full", "spearman_gate", "spearman_full"]
    per_target = (cells.groupby(["target", "model", "tau"], as_index=False)[qty]
                  .mean())

    
    pools = {"trees4": POOL_TREES,
             "all7": sorted(cells["model"].unique())}
    rows = []
    for pool, models in pools.items():
        sub = per_target[per_target["model"].isin(models)]
        g = sub.groupby(["target", "tau"], as_index=False)[qty].mean()
        g.insert(0, "pool", pool)
        g["n_models"] = [int((sub[(sub.target == r.target) & (sub.tau == r.tau)]
                              ).shape[0]) for r in g.itertuples()]
        rows.append(g)
    pooled = pd.concat(rows, ignore_index=True)

    cl = clusters.set_index("slug")
    agg = []
    for (pool, tau), g in pooled.groupby(["pool", "tau"]):
        for q in ("delta_spearman", "delta_kendall", "delta_spearman_ysd_matched"):
            v = g[q].to_numpy(dtype=float)
            keep = np.isfinite(v)
            v, sub = v[keep], g[keep]
            if len(v) < 2:
                continue
            sd, mean = float(v.std(ddof=1)), float(v.mean())
            base = {
                "pool": pool, "tau": tau, "quantity": q, "n_targets": len(v),
                "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum()),
                "sd_delta": sd,
                "T_needed": (int(np.ceil((1.96 * sd / mean) ** 2))
                             if mean != 0 else np.nan),
                "coverage_mean": float(g["coverage"].mean()),
                "is_headline": (pool, q.replace("delta_", ""), tau) == HEADLINE,
            }
            for unit, col in (("target", None), ("cluster_fine", "cluster_fine"),
                              ("cluster_coarse", "cluster_coarse")):
                units = (None if col is None
                         else np.array([cl.loc[t, col] for t in sub["target"]]))
                ci = _boot_ci(v, units, np.random.default_rng(BOOT_SEED))
                agg.append({**base, "boot_unit": unit, "n_units": ci["n_units"],
                            "mean": ci["delta_mean"],
                            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                            "ci_lo_cons": ci["ci_lo_cons"],
                            "ci_hi_cons": ci["ci_hi_cons"],
                            "ci_width": ci["ci_hi"] - ci["ci_lo"]})
    agg = pd.DataFrame(agg)

    
    widest = []
    for (pool, tau, q), g in agg.groupby(["pool", "tau", "quantity"]):
        w = g.loc[g["ci_width"].idxmax()]
        widest.append({"pool": pool, "tau": tau, "quantity": q,
                       "widest_unit": w["boot_unit"],
                       "mean": w["mean"], "ci_lo": w["ci_lo"], "ci_hi": w["ci_hi"],
                       "ci_lo_cons": w["ci_lo_cons"], "ci_hi_cons": w["ci_hi_cons"],
                       "n_targets": w["n_targets"], "n_pos": w["n_pos"],
                       "n_neg": w["n_neg"], "T_needed": w["T_needed"],
                       "coverage_mean": w["coverage_mean"],
                       "is_headline": w["is_headline"],
                       
                       "H8_1_supported": bool(w["ci_lo"] > 0)})
    return pooled, pd.merge(agg, pd.DataFrame(widest)[
        ["pool", "tau", "quantity", "widest_unit", "H8_1_supported"]],
        on=["pool", "tau", "quantity"], how="left"), pd.DataFrame(widest)


def main() -> int:
    ap = argparse.ArgumentParser(description="e24 8")
    ap.add_argument("--targets", nargs="*", default=None, help="default 20")
    args = ap.parse_args()

    prot = load_protocol()
    ad_cfg = prot.applicability_domain
    buckets = ad_cfg["buckets"]
    min_bucket_n = int(ad_cfg.get("min_bucket_n", 5))
    _assert_taus(buckets)
    assert_scope(MANIFEST)                      
    targets = args.targets or FROZEN20
    clusters = pd.read_csv(CLUSTERS, comment="#")

    print(f"[e24] target {len(targets)} protocol {PROTOCOL} τ={TAUS}"
          f"random B={B_RAND} threads {N_THREADS}")
    print(f"[e24] {DECLARED_IN}")

    rng = np.random.default_rng(RAND_SEED)
    all_cells, all_dropped, all_problems = [], [], []
    for i, slug in enumerate(targets, 1):
        t0 = time.time()
        c, d, p = run_target(slug, prot, buckets, min_bucket_n, rng)
        all_cells.extend(c)
        all_dropped.extend(d)
        all_problems.extend(p)
        print(f"[{i:2d}/{len(targets)}] {slug:<8} {len(c):3d}"
              f"{len(d):2d} {len(p)} {time.time() - t0:.0f}s")

    if all_problems:
        raise SystemExit("⛔ failed \n"
                         + "\n  ".join(all_problems[:40])
                         + (f"\n {len(all_problems)}" if len(all_problems) > 40
                            else ""))

    cells = pd.DataFrame(all_cells)
    dropped = pd.DataFrame(all_dropped) if all_dropped else pd.DataFrame(
        columns=["target", "fold", "model", "tau", "n_test", "n_keep", "reason"])
    per_target, agg, widest = aggregate(cells, clusters)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    save_table(cells, OUTDIR / "e24_per_cell.csv", float_fmt="%.8f")
    save_table(per_target, OUTDIR / "e24_per_target.csv", float_fmt="%.8f")
    save_table(agg, OUTDIR / "e24_agg_ci.csv", float_fmt="%.8f")
    save_table(widest, OUTDIR / "e24_widest_ci.csv", float_fmt="%.8f")
    save_table(dropped, OUTDIR / "e24_dropped_cells.csv", float_fmt="%.8f")
    
    cost = per_target[["pool", "target", "tau", "coverage", "delta_spearman",
                       "delta_spearman_ysd_matched", "y_sd_keep", "y_sd_full",
                       "spearman_full", "spearman_gate"]]
    save_table(cost, OUTDIR / "e24_coverage_cost.csv", float_fmt="%.8f")

    hl = widest[widest["is_headline"]]
    print("\n[e24] trees4 × spearman × τ=0.4 CI")
    if not hl.empty:
        r = hl.iloc[0]
        print(f"  Δ = {r['mean']:+.6f}  CI [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]"
              f"= {r['widest_unit']} CI"
              f"[{r['ci_lo_cons']:+.6f}, {r['ci_hi_cons']:+.6f}]")
        print(f"{int(r['n_pos'])}+ / {int(r['n_neg'])}−"
              f"mean {r['coverage_mean']:.4f}"
              f"{1 - r['coverage_mean']:.1%} molecules")
        if r["H8_1_supported"]:
            print("⇒ H8-1 §3.7")
        else:
            print(f"⇒ H8-1 **** §3.7"
                  f"20 target T_needed = {r['T_needed']} 90"
                  f"⛔ τ")
    print("\n[e24]")
    print(widest[["pool", "tau", "quantity", "widest_unit", "mean", "ci_lo",
                  "ci_hi", "n_pos", "n_neg", "T_needed", "coverage_mean",
                  "H8_1_supported"]].to_string(index=False))
    if not dropped.empty:
        print(f"\n[e24] {len(dropped)} §3.5")
        print(dropped.groupby(["tau", "target"]).size().to_string())

    write_run_meta(
        OUTDIR, target="cross_target", stage="e24_ad_gating",
        protocol_fingerprint=prot.fingerprint,
        extra={
            "declared_in": DECLARED_IN,
            "protocol": PROTOCOL,
            "taus": TAUS,
            "tau_source": "configs/protocol.yaml applicability_domain.buckets",
            "headline": {"pool": HEADLINE[0], "metric": HEADLINE[1], "tau": HEADLINE[2],
                         "unit": "fold mean modelmean"},
            "n_targets": len(targets),
            "n_cells": int(len(cells)),
            "n_dropped_cells": int(len(dropped)),
            "min_keep": MIN_KEEP,
            "b_rand": B_RAND, "rand_seed": RAND_SEED,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "conservative_inflation": CONSERVATIVE_INFLATION,
            "ysd_match": {"tol_rel": YSD_TOL, "min_accept": MATCH_MIN_ACCEPT,
                          "max_attempt": MATCH_MAX_ATTEMPT},
            "self_check": "p05 buckets.csv n spearman"
                          f"{SELF_CHECK_TOL} inconsistent",
            "not_done": [
                "predictions.csv y_pred %.4f n_tied_pairs_pred_keep"
                "%.4f prediction",
                "randomscaffold**** random test molecules"
                "p05 temporalresults",
                "no **** test"
                "****",
                "τ Lo-Hi 2023",
                "molecules bootstrap §11.3",
            ],
            "banned": [
                "target modelavailable",
                "message",
                "e24_coverage_cost.csv meanNN",
                "τ=0.6 / all7 / kendall",
            ],
        })
    print(f"\n[e24] write {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
