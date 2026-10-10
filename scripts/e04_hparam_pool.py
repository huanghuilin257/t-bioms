#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from pparbench.threads import apply_thread_limits              # noqa: E402

N_THREADS = apply_thread_limits()

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import yaml                                                     # noqa: E402
from sklearn.base import clone                                  # noqa: E402

from pparbench.config import load_protocol, load_target          # noqa: E402
from pparbench.io import (features_path, save_table,              # noqa: E402
                          target_results_dir, write_run_meta)
from pparbench.models import build_models, family_of             # noqa: E402
from pparbench.splits import (random_holdout_splits,             # noqa: E402
                              scaffold_kfold_splits, temporal_splits)

from e01_regret_analysis import METRICS, _metrics                # noqa: E402

GRID_PATH = ROOT / "configs" / "hparam_grid.yaml"
ALGOS = ["lightgbm", "hgb", "rf", "extratrees"]
PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]





IMPLICIT_DEFAULTS = {
    "hgb": {"min_samples_leaf": 20},
    "rf": {"max_depth": None},
    "extratrees": {"max_depth": None},
}


def _norm(d: dict) -> dict:
    """Documentation."""
    return {k: v for k, v in d.items() if v is not None}


def sample_configs(k: int, seed: int,
                   algos: list[str] | None = None) -> list[tuple[str, dict]]:
    """Documentation."""
    grid = yaml.safe_load(GRID_PATH.read_text(encoding="utf-8"))
    prot = load_protocol()
    rng = np.random.RandomState(seed)

    out: list[tuple[str, dict]] = []
    for algo in (algos or ALGOS):
        default = dict(prot.models[algo])
        default_eff = _norm({**IMPLICIT_DEFAULTS.get(algo, {}), **default})
        space = grid[algo]
        keys = sorted(space)
        combos = [dict(zip(keys, v)) for v in itertools.product(*(space[x] for x in keys))]
        combos = [c for c in combos if _norm({**default, **c}) != default_eff]
        if len(combos) < k - 1:
            raise RuntimeError(f"{algo} {len(combos)} default {k-1}")
        idx = rng.choice(len(combos), size=k - 1, replace=False)
        
        picked = [default] + [{**default, **combos[i]} for i in sorted(idx)]
        for i, params in enumerate(picked):
            out.append((f"{algo}#{i:02d}", params))
    return out


def build_folds(protocol: str, *, prot, tcfg, year, scaffold, n):
    cfg = prot.protocols[protocol]
    if protocol == "random_holdout":
        return list(random_holdout_splits(n, test_size=cfg["test_size"],
                                          seeds=cfg["seeds"]))
    if protocol == "scaffold_cv":
        return list(scaffold_kfold_splits(list(scaffold), n_splits=cfg["n_splits"],
                                          shuffle=cfg["shuffle"], seed=prot.seed))
    if protocol == "temporal":
        cuts = tcfg.cutpoints or cfg["default_cutpoints"]
        return list(temporal_splits(year, cuts, min_train=cfg["min_train"],
                                    min_test=cfg["min_test"]))
    raise ValueError(protocol)


def run(slug: str, k: int, algos: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prot = load_protocol()
    tcfg = load_target(slug)
    d = np.load(features_path(slug, "ligand2057"), allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    year, scaffold, mol_id = d["year"], d["scaffold"], d["mol_id"]
    n = len(y)

    configs = sample_configs(k, seed=prot.seed, algos=algos)
    print(f"[e04] {slug} n={n} {len(configs)} configuration × {len(PROTOCOLS)} protocol")
    print(f"[e04] threads: bagging n_jobs={prot.n_jobs}, boosting/OpenMP={N_THREADS}")

    fold_rows, pred_rows = [], []
    for protocol in PROTOCOLS:
        folds = build_folds(protocol, prot=prot, tcfg=tcfg, year=year,
                            scaffold=scaffold, n=n)
        if not folds:
            print(f"[warn] {protocol} novalid fold skip")
            continue
        print(f"\n[e04] === {protocol} === {len(folds)} folds")
        for cname, params in configs:
            algo = cname.split("#")[0]
            proto_model = build_models({algo: params}, seed=prot.seed,
                                       n_jobs=prot.n_jobs,
                                       boosting_n_jobs=prot.omp_num_threads,
                                       fp_bits=int(prot.features["morgan_bits"]))[algo]
            t0 = time.time()
            for fold_name, tr, te in folds:
                mdl = clone(proto_model)
                mdl.fit(X[tr], y[tr])
                pred = np.asarray(mdl.predict(X[te]), dtype=float)
                fold_rows.append({"protocol": protocol, "fold": fold_name,
                                  "model": cname, "algo": algo,
                                  "family": family_of(algo),
                                  "n_train": int(len(tr)), **_metrics(y[te], pred)})
                pred_rows.append(pd.DataFrame({
                    "protocol": protocol, "fold": fold_name, "model": cname,
                    "mol_id": mol_id[te], "y_true": y[te], "y_pred": pred}))
            print(f"[e04]   {cname:14s} {time.time()-t0:7.1f}s")
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def _pick(scores: pd.Series, pool: list[str]) -> tuple[str, int]:
    """Documentation."""
    avail = [m for m in pool if m in scores.index and np.isfinite(scores[m])]
    if not avail:
        return "", 0
    best = max(scores[m] for m in avail)
    ties = [m for m in avail if np.isclose(scores[m], best, atol=1e-12)]
    return ties[0], len(ties)


def analyse(slug: str, per_fold: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    agg = per_fold.groupby(["protocol", "model", "algo", "family"])[METRICS].mean()
    agg = agg.reset_index()
    rows = []
    for metric in METRICS:
        piv = agg.pivot_table(index="model", columns="protocol", values=metric)
        if "temporal" not in piv.columns:
            continue
        tmp = piv["temporal"].dropna()
        algo_of = dict(zip(agg["model"], agg["algo"]))
        fam_of = dict(zip(agg["model"], agg["family"]))

        full = list(tmp.index)
        tbest, _ = _pick(tmp, full)

        for selector in ("random_holdout", "scaffold_cv"):
            if selector not in piv.columns:
                continue
            sel = piv[selector].dropna()

            
            chosen, n_ties = _pick(sel, full)
            
            defaults = [m for m in full if m.endswith("#00")]
            chosen_def, _ = _pick(sel, defaults)

            
            boost_all = [m for m in full if fam_of[m] == "boosting"]
            tuned_boost, n_ties_boost = _pick(sel, boost_all)   
            rf_default = "rf#00"
            rows.append({
                "target": slug, "metric": metric, "selector": selector,
                "pool_size": len(full),
                # (a)
                "chosen_joint": chosen, "n_ties_joint": n_ties,
                "chosen_joint_temporal": float(tmp[chosen]),
                "temporal_best": tbest, "temporal_best_score": float(tmp[tbest]),
                "regret_joint": float(tmp[tbest] - tmp[chosen]),
                
                "chosen_default_pool": chosen_def,
                "regret_default_pool": float(tmp[tbest] - tmp[chosen_def]),
                # (b)
                "tuned_boosting": tuned_boost, "n_ties_boosting": n_ties_boost,
                "tuned_boosting_temporal": float(tmp[tuned_boost]),
                "rf_default_temporal": (float(tmp[rf_default])
                                        if rf_default in tmp.index else np.nan),
                "tuned_boosting_beats_default_rf": (
                    bool(tmp[tuned_boost] > tmp[rf_default])
                    if rf_default in tmp.index else None),
                "gap_rf_minus_tunedboost": (
                    float(tmp[rf_default] - tmp[tuned_boost])
                    if rf_default in tmp.index else np.nan),
                
                
                "best_boosting_temporal": (float(max(tmp[m] for m in boost_all))
                                           if boost_all else np.nan),
                "oracle_boosting_beats_default_rf": (
                    bool(max(tmp[m] for m in boost_all) > tmp[rf_default])
                    if boost_all and rf_default in tmp.index else None),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="e04 F")
    ap.add_argument("--target", required=True)
    ap.add_argument("--k", type=int, default=8, help="configuration default")
    ap.add_argument("--algos", nargs="*", default=ALGOS,
                    help="message"
                         "message")
    ap.add_argument("--reuse", action="store_true", help="skip")
    args = ap.parse_args()

    outdir = target_results_dir(args.target, "e04_hparam_pool")
    pf = outdir / "per_fold.csv"

    if args.reuse and pf.exists():
        per_fold = pd.read_csv(pf)
        print(f"[e04] {pf}")
    else:
        t0 = time.time()
        per_fold, preds = run(args.target, args.k, args.algos)
        save_table(per_fold, pf)
        save_table(preds, outdir / "predictions.csv")
        print(f"\n[e04] {time.time()-t0:.1f}s")

    
    ref = pd.read_csv(target_results_dir(args.target, "p03_protocol_matrix",
                                         create=False) / "per_fold.csv")
    ref = ref[ref["model"].isin(args.algos)][["protocol", "fold", "model", "spearman"]]
    mine = per_fold[per_fold["model"].str.endswith("#00")].copy()
    mine["model"] = mine["model"].str.replace("#00", "", regex=False)
    m = mine.merge(ref, on=["protocol", "fold", "model"], suffixes=("", "_p03"))
    if len(m):
        dmax = (m["spearman"] - m["spearman_p03"]).abs().max()
        print(f"[e04] default p03 {len(m)} {dmax:.6f}"
              f"{'OK ' if dmax < 1e-4 else '!! inconsistent default'}")

    res = analyse(args.target, per_fold)
    save_table(res, outdir / "regret.csv")

    print(f"\n===== {args.target} × =====")
    s = res[res["selector"] == "random_holdout"]
    print(s[["metric", "pool_size", "chosen_joint", "n_ties_joint",
             "chosen_joint_temporal", "temporal_best", "temporal_best_score",
             "regret_joint", "chosen_default_pool",
             "regret_default_pool"]].round(4).to_string(index=False))

    print("\n===== boosting rf =====")
    print(res[["metric", "selector", "tuned_boosting", "tuned_boosting_temporal",
               "rf_default_temporal", "gap_rf_minus_tunedboost",
               "tuned_boosting_beats_default_rf", "best_boosting_temporal",
               "oracle_boosting_beats_default_rf"]].round(4).to_string(index=False))

    write_run_meta(outdir, target=args.target, stage="e04_hparam_pool",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"k": args.k, "algos": args.algos,
                          "grid_file": str(GRID_PATH.relative_to(ROOT)),
                          "configs": [c for c, _ in sample_configs(
                              args.k, load_protocol().seed, args.algos)],
                          "notes": "note_8/04 F configs/hparam_grid.yaml"
                                   "results protocol"})
    print(f"\n[e04] write {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
