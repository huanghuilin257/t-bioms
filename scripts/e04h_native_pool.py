#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import e04_level_dissociation as ld                               # noqa: E402
from pparbench.io import save_table, write_run_meta               # noqa: E402

OUTDIR = ROOT / "results" / "cross_target" / "e04h_native_pool"


def analyse_target_native(slug: str, agg: pd.DataFrame, *,
                          algos: list[str], families: dict,
                          with_l3: bool) -> list[dict]:
    """Documentation."""
    algo_of = dict(zip(agg["model"], agg["algo"]))
    fam_of = dict(zip(agg["model"], agg["family"]))
    rows = []
    for metric in ld.METRICS:
        piv = agg.pivot_table(index="model", columns="protocol", values=metric)
        if ld.EVAL_PROTOCOL not in piv.columns:
            continue
        temporal = piv[ld.EVAL_PROTOCOL].dropna()
        for selector in ld.SELECTORS:
            if selector not in piv.columns:
                continue
            sel = piv[selector].dropna()
            pools = ld.build_pools(sel, algo_of, fam_of, algos, families, with_l3)
            for level, named in pools.items():
                per_pool = []
                for pname, members in named.items():
                    members = [m for m in members if m in temporal.index]
                    if not members:
                        continue
                    st = ld.subpool_stats(temporal, sel, members, len(members))
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
                    "level": level, "k": "native",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-l3", action="store_true")
    ap.add_argument("--exclude-gin", action="store_true")
    ap.add_argument("--gnn-agg", default="seed_median",
                    choices=["seed_median", "fold_mean_only"])
    ap.add_argument("--e04g-dir", default="e04g_hparam_pool_gnn_merged")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ld.E04G_DIR = args.e04g_dir
    ok_gnn, missing, _ = ld.e04g_complete(verbose=args.allow_l3)
    with_l3 = False
    if args.allow_l3:
        if not ok_gnn:
            print(f"[e04h] ⛔ L3 e04g {len(missing)}/20 -> {missing}")
            return 2
        with_l3 = True

    families = dict(ld.TREE_FAMILIES)
    if with_l3:
        gnn = [a for a in ld.GNN_ALGOS if not (args.exclude_gin and a == "gin")]
        families["gnn"] = gnn
    algos = [a for fam in families.values() for a in fam]
    scope = ("trees_only_4algo_2family" if not with_l3
             else f"full_{len(algos)}algo_{len(families)}family"
                  f"{'_no_gin' if args.exclude_gin else '_with_gin'}"
                  f"__{args.gnn_agg}")
    print(f"[e04h] = {scope} algos = {algos}")

    rows, used = [], []
    for slug in ld.FROZEN20:
        try:
            agg, diag = ld.build_agg(slug, include_gnn=with_l3,
                                     gnn_agg=args.gnn_agg,
                                     exclude_gin=args.exclude_gin)
        except FileNotFoundError as e:
            print(f"[e04h] ⚠ {slug} {e} skip")
            continue
        if diag["n_nan_unexplained"]:
            print(f"[e04h] ⛔ {slug} {diag['n_nan_unexplained']} NaN")
            return 1
        pool_models = agg[agg.algo.isin(algos)]
        if pool_models[ld.METRICS].isna().any().any():
            print(f"[e04h] ⛔ {slug} NaN")
            return 1
        rows += analyse_target_native(slug, agg, algos=algos,
                                      families=families, with_l3=with_l3)
        used.append(slug)

    per_target = pd.DataFrame(rows)
    if per_target.empty:
        print("[e04h] ⛔ noavailabledata")
        return 1
    print(f"[e04h] target {len(used)}/20 per_target {len(per_target)}")
    print("[e04h]")
    print(per_target.groupby("level")["native_pool_size"]
          .agg(["min", "max", "nunique"]).to_string())

    cl = pd.read_csv(ld.CLUSTERS, comment="#").set_index("slug")
    absent = sorted(set(used) - set(cl.index))
    if absent:
        print(f"[e04h] ⛔ target resample {absent}")
        return 1
    summary, signs = ld.summarise(per_target, cl)

    outdir = OUTDIR / (args.tag or scope)
    outdir.mkdir(parents=True, exist_ok=True)
    save_table(per_target, outdir / "per_target_level.csv", float_fmt="%.6f")
    save_table(summary, outdir / "level_summary.csv", float_fmt="%.6f")
    save_table(signs, outdir / "level_sign_tests.csv", float_fmt="%.6f")
    
    per_target.to_csv(outdir / "per_target_level__fullprec.csv", index=False)
    summary.to_csv(outdir / "level_summary__fullprec.csv", index=False)
    signs.to_csv(outdir / "level_sign_tests__fullprec.csv", index=False)

    write_run_meta(
        outdir, target="cross_target", stage="e04h_native_pool",
        protocol_fingerprint=f"selectors={ld.SELECTORS};eval={ld.EVAL_PROTOCOL}",
        extra={
            "declared_in": ld.DECLARED_IN,
            "scope": scope,
            "with_l3": with_l3,
            "k": "native = ⛔",
            "native_pool_size_observed": {
                lv: sorted(g["native_pool_size"].unique().tolist())
                for lv, g in per_target.groupby("level")},
            "n_targets": len(used),
            "reuses": "scripts/e04_level_dissociation.py build_agg / build_pools /"
                      "subpool_stats / _pick / widest_ci / summarise ⛔",
            "only_difference": "analyse_target k (2, 4) len(members)"
                               "message",
            "not_comparable_with": "M7.3 32 configuration /"
                                   "⛔ note_8/37 M7.4 4",
            "size_matched": False,
            "size_matched_note": "**** ⇒ not size-matched"
                                 "note_8/43:62",
            "sign_tests_caveat": "L1=8 L2=4/6/7"
                                 "L3=3 ⇒ ⛔ k=2",
            "gnn_agg": args.gnn_agg,
            "exclude_gin": args.exclude_gin,
            "e04g_dir": args.e04g_dir,
            "precision": "*.csv save_table %.6f"
                         "*__fullprec.csv to_csv",
        })
    print(f"[e04h] ✅ write {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
