from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from pparbench.ad import (bucket_metrics, coverage, monotonicity_check,
                          nearest_neighbour_similarity)
from pparbench.config import load_protocol, load_target
from pparbench.io import (features_path, save_table, target_data_dir,
                          target_results_dir, write_run_meta)


def load_ad_fingerprints(slug: str) -> list:
    """Documentation."""
    d = np.load(target_data_dir(slug) / "features_ad.npz", allow_pickle=True)
    bits = d["bits"]
    fps = []
    for row in bits:
        bv = ExplicitBitVect(bits.shape[1])
        on = np.flatnonzero(row).tolist()
        if not on:
            fps.append(None)      
            continue
        bv.SetBitsFromList(on)
        fps.append(bv)
    return fps


def main() -> int:
    ap = argparse.ArgumentParser(description="applicability domain temporalprotocol")
    ap.add_argument("--target", required=True)
    ap.add_argument("--protocol", default="temporal")
    args = ap.parse_args()

    prot = load_protocol()
    tcfg = load_target(args.target)
    ad_cfg = prot.applicability_domain
    buckets = ad_cfg["buckets"]
    min_bucket_n = int(ad_cfg.get("min_bucket_n", 5))
    thr = float(buckets[0][1])          

    pred_path = target_results_dir(tcfg.slug, "p03_protocol_matrix") / "predictions.csv"
    if not pred_path.exists():
        print(f"[error] missing {pred_path} run first p03")
        return 1
    pred = pd.read_csv(pred_path)
    pred = pred[pred["protocol"] == args.protocol].copy()
    if pred.empty:
        print(f"[error] predictions.csv no protocol={args.protocol}")
        return 1

    fps = load_ad_fingerprints(tcfg.slug)
    d = np.load(features_path(tcfg.slug, "ligand2057"), allow_pickle=True)
    year = d["year"]
    n = len(year)

    print(f"[p05] target={tcfg.slug}  protocol={args.protocol}  "
          f"={buckets} ={thr}")

    cov_rows, bucket_rows, mono_rows = [], [], []

    for fold, g_fold in pred.groupby("fold"):
        
        test_idx = np.unique(g_fold["test_idx"].to_numpy())
        if args.protocol == "temporal" and str(fold).startswith("T"):
            T = int(str(fold)[1:])
            train_idx = np.flatnonzero(~np.isnan(year) & (year < T))
        else:
            train_idx = np.setdiff1d(np.arange(n), test_idx)

        sims_test = nearest_neighbour_similarity(fps, train_idx, test_idx)
        sim_by_idx = dict(zip(test_idx.tolist(), sims_test.tolist()))

        cov = coverage(sims_test, threshold=thr)
        cov_rows.append({"fold": fold, "n_train": len(train_idx),
                         "n_test": len(test_idx), **cov})
        print(f"\n[p05] {fold}: train={len(train_idx)} test={len(test_idx)}  "
              f"meanNN={cov['mean_nn_sim']:.3f}  "
              f"(<{thr})={cov['out_of_domain_frac']:.1%}")

        for model, g in g_fold.groupby("model"):
            s = g["test_idx"].map(sim_by_idx).to_numpy(dtype=float)
            bm = bucket_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), s,
                                buckets, min_n=min_bucket_n)
            bm.insert(0, "model", model)
            bm.insert(0, "fold", fold)
            bucket_rows.append(bm)

            mono = monotonicity_check(bm)
            mono_rows.append({"fold": fold, "model": model,
                              "n_buckets_evaluable": mono["n_buckets"],
                              "spearman_by_bucket": ";".join(
                                  f"{v:.3f}" for v in mono["values"]),
                              "monotone_increasing": mono["monotone_increasing"],
                              "endpoints_increasing": mono["endpoints_increasing"],
                              "endpoint_gap": mono["endpoint_gap"],
                              
                              
                              "n_high_sim_bucket": int(bm["n"].iloc[-1]),
                              "y_std_high_sim_bucket": float(bm["y_std"].iloc[-1])})
            line = "  ".join(
                f"{r.bucket}:n={r.n},Sp="
                + ("nan" if pd.isna(r.spearman) else f"{r.spearman:.3f}")
                + (",σy=" + ("nan" if pd.isna(r.y_std) else f"{r.y_std:.2f}"))
                for r in bm.itertuples())
            print(f"{model:<11} {line} ={mono['monotone_increasing']}"
                  f"={mono['endpoints_increasing']}"
                  f"({mono['endpoint_gap']:+.3f})")

    outdir = target_results_dir(tcfg.slug, "p05_ad")
    cov_df = pd.DataFrame(cov_rows)
    bucket_df = pd.concat(bucket_rows, ignore_index=True)
    mono_df = pd.DataFrame(mono_rows)
    save_table(cov_df, outdir / "coverage.csv")
    save_table(bucket_df, outdir / "buckets.csv")
    save_table(mono_df, outdir / "monotonicity.csv")

    n_mono = int(mono_df["monotone_increasing"].sum())
    print(f"\n[p05] (fold,model) : {n_mono}/{len(mono_df)}")
    if n_mono == len(mono_df):
        print("⇒ AD")
    elif n_mono >= 0.75 * len(mono_df):
        print("⇒")
    else:
        print("⇒ ③")

    
    
    fail = mono_df[~mono_df["monotone_increasing"]]
    if len(fail):
        print(f"[p05] fold: {sorted(fail['fold'].unique())}")
        print(f"fold n median ="
              f"{fail['n_high_sim_bucket'].median():.0f}，"
              f"σy median = {fail['y_std_high_sim_bucket'].median():.2f}")
        ok = mono_df[mono_df["monotone_increasing"]]
        if len(ok):
            print(f"fold n median ="
                  f"{ok['n_high_sim_bucket'].median():.0f}，"
                  f"σy median = {ok['y_std_high_sim_bucket'].median():.2f}")
        print("message"
              "samples")

    
    n_end = int(mono_df["endpoints_increasing"].sum())
    print(f"[p05] > (fold,model) : {n_end}/{len(mono_df)}")
    end_fail = mono_df[~mono_df["endpoints_increasing"]]
    if len(end_fail):
        print(f"failed fold {sorted(end_fail['fold'].unique())}"
              f"n ="
              f"{sorted(end_fail['n_high_sim_bucket'].unique())}")
        end_ok = mono_df[mono_df["endpoints_increasing"]]
        print(f"fold n ="
              f"{int(end_ok['n_high_sim_bucket'].min())}")

    
    min_report_n = int(ad_cfg.get("min_report_n", 100))
    reportable = mono_df[mono_df["n_high_sim_bucket"] >= min_report_n]
    if len(reportable):
        n_ok = int(reportable["monotone_increasing"].sum())
        n_ok_end = int(reportable["endpoints_increasing"].sum())
        folds_ok = sorted(reportable["fold"].unique())
        print(f"[p05] n>={min_report_n} :"
              f"{n_ok}/{len(reportable)}"
              f"{n_ok_end}/{len(reportable)} fold {folds_ok}")
        print("fold")
    else:
        print(f"[p05] fold n>={min_report_n}"
              f"targetsupported ③ data")

    
    if args.protocol == "temporal":
        c = cov_df.copy()
        c["T"] = c["fold"].astype(str).str.lstrip("T").astype(int)
        c = c.sort_values("T")
        print("\n[p05] meanNN :" +
              " → ".join(f"{r.T}:{r.mean_nn_sim:.3f}" for r in c.itertuples()))
        m = c["mean_nn_sim"].to_numpy()
        decreasing = bool(np.all(np.diff(m) < 0))
        print(f"= {m[0] - m[-1]:+.3f} = {decreasing}")
        if not decreasing:
            print("⚠"
                  "message"
                  f"{cov_df['out_of_domain_frac'].min():.0%}–"
                  f"{cov_df['out_of_domain_frac'].max():.0%} molecules")

    write_run_meta(
        outdir, target=tcfg.slug, stage="p05_ad",
        protocol_fingerprint=prot.fingerprint,
        extra={"protocol": args.protocol, "buckets": buckets,
               "ood_threshold": thr,
               "n_monotone": n_mono, "n_combinations": len(mono_df),
               "min_report_n": min_report_n,
               "n_endpoints_increasing": n_end,
               "n_monotone_reportable": int(reportable["monotone_increasing"].sum()),
               "n_endpoints_increasing_reportable": int(reportable["endpoints_increasing"].sum()),
               "n_combinations_reportable": len(reportable),
               "mean_ood_frac": float(cov_df["out_of_domain_frac"].mean())},
    )
    print(f"\n[p05] results {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
