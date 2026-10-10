from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



from pparbench.threads import apply_thread_limits

N_THREADS = apply_thread_limits()

import numpy as np
import pandas as pd

from pparbench.config import load_protocol, load_target
from pparbench.io import (features_path, save_table, target_results_dir,
                          write_run_meta)
from pparbench.metrics import aggregate_folds, compute_metrics
from pparbench.models import build_models, family_of
from pparbench.splits import (random_holdout_splits, scaffold_kfold_splits,
                              temporal_splits)

PROTOCOL_ORDER = ["random_holdout", "scaffold_cv", "temporal"]


def build_folds(protocol: str, *, prot, tcfg, year, scaffold, n):
    """Documentation."""
    cfg = prot.protocols[protocol]
    if protocol == "random_holdout":
        return list(random_holdout_splits(n, test_size=cfg["test_size"], seeds=cfg["seeds"]))
    if protocol == "scaffold_cv":
        return list(scaffold_kfold_splits(list(scaffold), cfg["n_splits"],
                                          seed=prot.seed, shuffle=cfg["shuffle"]))
    if protocol == "temporal":
        cuts = tcfg.cutpoints or cfg["default_cutpoints"]
        return list(temporal_splits(year, cuts,
                                    min_train=cfg["min_train"], min_test=cfg["min_test"]))
    raise ValueError(protocol)


def main() -> int:
    ap = argparse.ArgumentParser(description="protocol × model")
    ap.add_argument("--target", required=True)
    ap.add_argument("--protocols", nargs="*", default=PROTOCOL_ORDER,
                    help=f"running all by default: {PROTOCOL_ORDER}")
    ap.add_argument("--models", nargs="*", default=None, help="default protocol.yaml")
    args = ap.parse_args()

    prot = load_protocol()
    tcfg = load_target(args.target)

    fp = features_path(tcfg.slug, "ligand2057")
    if not fp.exists():
        print(f"[error] missing {fp} run first p02_features.py --target {tcfg.slug}")
        return 1
    d = np.load(fp, allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    year, scaffold, mol_id = d["year"], d["scaffold"], d["mol_id"]
    n = len(y)

    models = build_models(prot.models, seed=prot.seed, n_jobs=prot.n_jobs,
                          boosting_n_jobs=prot.omp_num_threads,
                          fp_bits=int(prot.features["morgan_bits"]))
    if args.models:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            print(f"[error] unknownmodel {unknown} available: {list(models)}")
            return 1
        models = {k: v for k, v in models.items() if k in args.models}

    enrich = prot.metrics.get("enrichment_pct", [])
    print(f"[p03] target={tcfg.slug}  n={n}  X={X.shape}")
    print(f"[p03] threads: bagging n_jobs={prot.n_jobs}, boosting/OpenMP={N_THREADS}")
    print(f"[p03] model: {list(models)}")
    print(f"[p03] protocol: {args.protocols}")

    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    agg_rows: list[dict] = []

    for protocol in args.protocols:
        folds = build_folds(protocol, prot=prot, tcfg=tcfg, year=year,
                            scaffold=scaffold, n=n)
        if not folds:
            print(f"[warn] {protocol} valid fold skip")
            continue
        print(f"\n[p03] === {protocol} === {len(folds)} folds: "
              f"{[f'{f[0]}(tr{len(f[1])}/te{len(f[2])})' for f in folds]}")

        for mname, proto_model in models.items():
            per_fold, pooled_t, pooled_p = [], [], []
            t0 = time.time()
            for fold_name, tr, te in folds:
                from sklearn.base import clone
                mdl = clone(proto_model)
                mdl.fit(X[tr], y[tr])
                pred = np.asarray(mdl.predict(X[te]), dtype=float)

                met = compute_metrics(y[te], pred, enrichment_pct=enrich)
                per_fold.append(met)
                fold_rows.append({
                    "protocol": protocol, "fold": fold_name, "model": mname,
                    "family": family_of(mname),
                    "n_train": int(len(tr)), **met,
                })
                pooled_t.append(y[te]); pooled_p.append(pred)
                pred_rows.append(pd.DataFrame({
                    "protocol": protocol, "fold": fold_name, "model": mname,
                    "mol_id": mol_id[te], "test_idx": te,
                    "y_true": y[te], "y_pred": pred,
                }))

            pt, pp = np.concatenate(pooled_t), np.concatenate(pooled_p)
            agg = aggregate_folds(per_fold, pt, pp, enrichment_pct=enrich)
            agg_rows.append({"protocol": protocol, "model": mname,
                             "family": family_of(mname), **agg,
                             "seconds": round(time.time() - t0, 1)})
            print(f"  {mname:<11} fold-mean Sp={agg['spearman_foldmean']:.3f}"
                  f"±{agg['spearman_foldstd']:.3f}   pooled Sp={agg['spearman_pooled']:.3f}"
                  f"   ({time.time()-t0:.0f}s)")

    outdir = target_results_dir(tcfg.slug, "p03_protocol_matrix")
    per_fold_df = pd.DataFrame(fold_rows)
    agg_df = pd.DataFrame(agg_rows)
    save_table(per_fold_df, outdir / "per_fold.csv")
    save_table(agg_df, outdir / "aggregated.csv")
    save_table(pd.concat(pred_rows, ignore_index=True), outdir / "predictions.csv")

    
    rank_rows = []
    for protocol, g in agg_df.groupby("protocol"):
        for key, label in (("spearman_foldmean", "fold_mean"),
                           ("spearman_pooled", "pooled")):
            s = g.sort_values(key, ascending=False)
            for r, (_, row) in enumerate(s.iterrows(), start=1):
                rank_rows.append({"protocol": protocol, "aggregation": label,
                                  "rank": r, "model": row["model"],
                                  "family": row["family"], "spearman": row[key]})
    rank_df = pd.DataFrame(rank_rows)
    save_table(rank_df, outdir / "ranking.csv")

    
    print("\n[p03] protocol fold-mean")
    winners = {}
    for protocol, g in rank_df[rank_df.aggregation == "fold_mean"].groupby("protocol"):
        w = g[g["rank"] == 1].iloc[0]
        winners[protocol] = (w["model"], w["family"])
        print(f"  {protocol:<15} → {w['model']} ({w['family']}, Sp={w['spearman']:.3f})")
    fams = {v[1] for v in winners.values()}
    if len(fams) > 1:
        print("⇒ model family changes by protocol ②")
    else:
        print("⇒ protocol ② per_fold")

    print("\n[p03] fold-mean vs pooled consistent")
    for protocol in rank_df["protocol"].unique():
        sub = rank_df[(rank_df.protocol == protocol) & (rank_df["rank"] == 1)]
        ms = sub.set_index("aggregation")["model"].to_dict()
        flag = "consistent" if len(set(ms.values())) == 1 else "★inconsistent (evidence layer 1)"
        print(f"  {protocol:<15} fold_mean={ms.get('fold_mean')} "
              f"pooled={ms.get('pooled')}  → {flag}")

    write_run_meta(
        outdir, target=tcfg.slug, stage="p03_protocol_matrix",
        protocol_fingerprint=prot.fingerprint,
        extra={"n_molecules": n, "models": list(models),
               "protocols": args.protocols,
               "foldmean_winners": {k: v[0] for k, v in winners.items()}},
    )
    print(f"\n[p03] results {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
