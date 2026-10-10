#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/no_such_mps_dir_pafb")
warnings.filterwarnings("ignore")

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import yaml                                                          # noqa: E402

from pparbench.config import RESULTS_DIR, load_protocol, load_target  # noqa: E402
from pparbench.io import features_path                                # noqa: E402
from pparbench.metrics import compute_metrics                         # noqa: E402

from pparbench.splits import (random_holdout_splits,                  # noqa: E402
                              scaffold_kfold_splits, temporal_splits)

PROTOCOL_ORDER = ["random_holdout", "scaffold_cv", "temporal"]
OUT_DIR = RESULTS_DIR / "r2_gnn"        
GRID_FILE = ROOT / "configs" / "hparam_grid_r2_gnn.yaml"


def build_folds(protocol: str, *, prot, tcfg, year, scaffold, n):
    """Documentation."""
    cfg = prot.protocols[protocol]
    if protocol == "random_holdout":
        return list(random_holdout_splits(n, test_size=cfg["test_size"],
                                          seeds=cfg["seeds"]))
    if protocol == "scaffold_cv":
        return list(scaffold_kfold_splits(list(scaffold), cfg["n_splits"],
                                          seed=prot.seed, shuffle=cfg["shuffle"]))
    if protocol == "temporal":
        cuts = tcfg.cutpoints or cfg["default_cutpoints"]
        return list(temporal_splits(year, cuts, min_train=cfg["min_train"],
                                    min_test=cfg["min_test"]))
    raise ValueError(protocol)


def load_target_arrays(slug: str):
    """Documentation."""
    fp = features_path(slug, "ligand2057")
    if not fp.exists():
        raise FileNotFoundError(f"{fp} run first p02_features.py --target {slug}")
    d = np.load(fp, allow_pickle=True)
    y = d["y"].astype(np.float64)
    csv = pd.read_csv(ROOT / "data" / slug / "dataset.csv")
    npz_ids = [str(x) for x in d["mol_id"]]
    csv_ids = [str(x) for x in csv["mol_id"]]
    if npz_ids != csv_ids:
        raise RuntimeError(
            f"{slug}: npz dataset.csv mol_id inconsistent"
            "prediction y_true failed")
    if not np.allclose(y, csv["y"].to_numpy(dtype=float), equal_nan=True):
        raise RuntimeError(f"{slug}: npz y dataset.csv y inconsistent")
    return y, d["year"], d["scaffold"], np.asarray(d["mol_id"]),\
        csv["smiles"].to_numpy(dtype=object)


def append_rows(path: Path, rows: list[dict]) -> None:
    """Documentation."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, header=header)
        fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="GNN")
    ap.add_argument("--targets", nargs="*", default=["pparg"],
                    help="target slug all")
    ap.add_argument("--models", nargs="*", default=None,
                    help="default GNN pparbench.gnn.GNN_NAMES")
    ap.add_argument("--protocols", nargs="*", default=PROTOCOL_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="default seeds")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--tag", default="", help="output")
    ap.add_argument("--out-dir", default="r2_gnn",
                    help="results/ output ****"
                         "r2_gnn prediction r2_gnn_v2")
    args = ap.parse_args()

    
    
    from pparbench.gnn import GNN_NAMES, build_gnn, enable_determinism

    enable_determinism()   

    models = args.models if args.models else list(GNN_NAMES)
    unknown = [m for m in models if m not in GNN_NAMES]
    if unknown:
        print(f"[error] unknown GNN {unknown} available: {list(GNN_NAMES)}")
        return 1

    grid_cfg = yaml.safe_load(GRID_FILE.read_text(encoding="utf-8"))
    defaults = grid_cfg["defaults"]
    seeds = args.seeds if args.seeds else [0, 1, 2]
    prot = load_protocol()
    enrich = prot.metrics.get("enrichment_pct", [])

    if args.targets == ["all"]:
        
        man = pd.read_csv(ROOT / "configs" / "manifests" / "targets_manifest.csv", comment="#")
        targets = sorted(man["slug"].astype(str).tolist())
    else:
        targets = list(args.targets)
    targets = [t for i, t in enumerate(targets) if i % args.n_shards == args.shard]

    out_dir = RESULTS_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    fold_path = out_dir / f"r2_per_fold{suffix}.csv"
    pred_path = out_dir / f"r2_predictions{suffix}.csv"
    skip_path = out_dir / f"r2_skipped{suffix}.csv"

    print(f"[r2] target {targets}", flush=True)
    print(f"[r2] model {models} | protocol {args.protocols} | {seeds}", flush=True)
    print(f"[r2] : {grid_cfg['early_stopping']} §6 epochs",
          flush=True)
    print(f"[r2] {fold_path}", flush=True)
    print(f"[r2] moleculesprediction {pred_path} kendall/pooled/ bootstrap",
          flush=True)
    print(f"[r2] : enable_determinism + Trainer(deterministic=True)"
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)

    for slug in targets:
        try:
            tcfg = load_target(slug)
            y, year, scaffold, mol_id, smiles = load_target_arrays(slug)
        except Exception as exc:                                   # noqa: BLE001
            
            print(f"[r2] !! {slug} : {type(exc).__name__}: {exc}", flush=True)
            append_rows(skip_path, [{"target": slug, "reason": f"{type(exc).__name__}: {exc}"}])
            continue
        n = len(y)
        print(f"\n[r2] ===== {slug} n={n} =====", flush=True)

        for protocol in args.protocols:
            folds = build_folds(protocol, prot=prot, tcfg=tcfg, year=year,
                                scaffold=scaffold, n=n)
            if not folds:
                print(f"[r2] {slug}/{protocol} valid fold skip", flush=True)
                append_rows(skip_path, [{"target": slug, "reason": f"{protocol}: valid fold"}])
                continue
            print(f"[r2] {slug}/{protocol}: {len(folds)} folds "
                  f"{[f'{f[0]}(tr{len(f[1])}/te{len(f[2])})' for f in folds]}", flush=True)

            for mname in models:
                params = dict(defaults[mname])
                rows: list[dict] = []
                prows: list[dict] = []
                t0 = time.time()
                for fold_name, tr, te in folds:
                    for seed in seeds:
                        try:
                            mdl = build_gnn(mname, params, seed=seed, device="cuda")
                            mdl.fit(smiles[tr], y[tr])
                            pred = np.asarray(mdl.predict(smiles[te]), dtype=float)
                        except Exception as exc:                    # noqa: BLE001
                            print(f"[r2] !! {slug}/{protocol}/{mname}/{fold_name}/s{seed} "
                                  f"failed: {type(exc).__name__}: {exc}", flush=True)
                            append_rows(skip_path, [{
                                "target": slug, "reason": f"{protocol}/{mname}/{fold_name}"
                                                          f"/s{seed}: {type(exc).__name__}: {exc}"}])
                            continue
                        ok = np.isfinite(pred)
                        
                        met = compute_metrics(y[te][ok], pred[ok], enrichment_pct=enrich)
                        rows.append({
                            "target": slug, "protocol": protocol, "model": mname,
                            "fold": fold_name, "seed": seed,
                            "n_train": int(len(tr)), "n_test": int(ok.sum()),
                            "n_pred_nan": int((~ok).sum()), **met,
                        })
                        
                        
                        
                        for j, ti in enumerate(te):
                            prows.append({
                                "target": slug, "protocol": protocol, "model": mname,
                                "fold": fold_name, "seed": seed,
                                "mol_id": mol_id[ti], "test_idx": int(ti),
                                "y_true": float(y[ti]), "y_pred": float(pred[j]),
                            })
                append_rows(fold_path, rows)
                append_rows(pred_path, prows)
                if rows:
                    sp = np.median([r.get("spearman", np.nan) for r in rows])
                    print(f"[r2]   {mname:12s} {time.time() - t0:7.1f}s  "
                          f"{len(rows)} rows spearman(×fold median)={sp:+.4f}", flush=True)

    print(f"\n[r2] {fold_path}", flush=True)
    print("[r2] mixed7 modelconsistent bootstrap",
          flush=True)
    
    
    
    
    if skip_path.exists():
        n_skip = max(sum(1 for _ in skip_path.open(encoding="utf-8")) - 1, 0)
        if n_skip:
            print(f"[r2] !! {skip_path.name} {n_skip} skip results"
                  f"message", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
