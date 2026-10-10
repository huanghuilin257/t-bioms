#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits                  # noqa: E402

apply_thread_limits()                                              

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import yaml                                                        # noqa: E402

from pparbench.config import load_protocol, load_target            # noqa: E402
from pparbench.metrics import compute_metrics                      # noqa: E402
from pparbench.io import write_run_meta                        # noqa: E402
from r2_gnn_matrix import build_folds, load_target_arrays          # noqa: E402

GRID_PATH = ROOT / "configs" / "hparam_grid_r2_gnn.yaml"
PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]
GNN_FAMILY = "gnn"


def sample_configs(k: int, seed: int,
                   models: list[str] | None = None) -> list[tuple[str, str, dict]]:
    """Documentation."""
    cfg = yaml.safe_load(GRID_PATH.read_text(encoding="utf-8"))
    defaults, grid = cfg["defaults"], cfg["grid"]
    rng = np.random.RandomState(seed)

    out: list[tuple[str, str, dict]] = []
    for name in (models or sorted(grid)):
        default, space = dict(defaults[name]), grid[name]
        missing = [x for x in space if x not in default]
        if missing:
            
            raise RuntimeError(f"{name}: {missing} defaults")
        keys = sorted(space)
        combos = [dict(zip(keys, v)) for v in itertools.product(*(space[x] for x in keys))]
        
        combos = [c for c in combos if any(default[x] != c[x] for x in keys)]
        if len(combos) < k - 1:
            raise RuntimeError(f"{name} {len(combos)} default {k - 1}")
        idx = rng.choice(len(combos), size=k - 1, replace=False)
        picked = [default] + [{**default, **combos[i]} for i in sorted(idx)]
        for i, params in enumerate(picked):
            out.append((f"{name}#{i:02d}", name, params))
    return out


def append_rows(path: Path, rows: list[dict]) -> None:
    """Documentation."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    with path.open("a", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, header=not path.exists() or path.stat().st_size == 0)
        fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="GNN e04")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="manifests/targets_manifest.csv target")
    ap.add_argument("--k", type=int, default=8, help="GNN configuration §9 8")
    ap.add_argument("--models", nargs="*", default=None,
                    help="⛔ **** sample_configs")
    ap.add_argument("--only-cells", default=None, metavar="CSV",
                    help="target,protocol,model,fold,seed"
                         "scripts/e04g_topup_manifest.py")
    ap.add_argument("--protocols", nargs="*", default=PROTOCOLS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-name", default="e04g_hparam_pool_gnn")
    args = ap.parse_args()

    if args.k != 8:
        print(f"[e04g] ⚠ K={args.k} §9 8 smoke test"
              f"message", flush=True)

    only = None
    if args.only_cells:
        oc = pd.read_csv(args.only_cells)
        need = {"target", "protocol", "model", "fold", "seed"}
        if not need <= set(oc.columns):
            ap.error(f"--only-cells missing columns: {sorted(need - set(oc.columns))}")
        only = set(zip(oc["target"], oc["protocol"], oc["model"],
                       oc["fold"], oc["seed"].astype(int)))
        print(f"[e04g] --only-cells: {len(only)}"
              f"| {sorted({t for t, *_ in only})}", flush=True)

    if args.all:
        
        
        
        
        
        
        from e22_scale_rollups import _frozen20  # noqa: E402
        targets = _frozen20()
    elif args.targets:
        targets = args.targets
    elif only is not None:
        targets = sorted({t for t, *_ in only})      
    else:
        ap.error("--targets --all --only-cells")
    targets = [t for i, t in enumerate(sorted(targets)) if i % args.n_shards == args.shard]

    from pparbench.gnn import build_gnn, enable_determinism
    enable_determinism()

    prot = load_protocol()
    
    
    
    
    
    
    
    
    
    configs = sample_configs(args.k, seed=prot.seed, models=None)
    if args.models:
        keep = set(args.models)
        unknown = keep - {m for _, m, _ in configs}
        if unknown:
            ap.error(f"--models : {sorted(unknown)}")
        configs = [c for c in configs if c[1] in keep]

    enrich = prot.report.get("enrichment_pct", [5, 10]) if hasattr(prot, "report") else [5, 10]

    print(f"[e04g] {len(targets)} : {targets}", flush=True)
    print(f"[e04g] {len(configs)} configuration = {len(set(c[1] for c in configs))} model × K={args.k}"
          f"| protocol {args.protocols} | {args.seeds} | {args.device}", flush=True)
    print(f"[e04g] ⛔ configuration results = results"
          f"failed skipped.csv", flush=True)

    for slug in targets:
        outdir = ROOT / "results" / slug / args.out_name
        outdir.mkdir(parents=True, exist_ok=True)
        pf_path, pred_path = outdir / "per_fold.csv", outdir / "predictions.csv"
        skip_path = outdir / "skipped.csv"
        try:
            tcfg = load_target(slug)
            y, year, scaffold, mol_id, smiles = load_target_arrays(slug)
        except Exception as exc:                                   # noqa: BLE001
            
            print(f"[e04g] !! {slug} : {type(exc).__name__}: {exc}", flush=True)
            append_rows(skip_path, [{"target": slug,
                                     "reason": f"{type(exc).__name__}: {exc}"}])
            continue
        n = len(y)
        print(f"\n[e04g] ===== {slug} n={n} =====", flush=True)

        for protocol in args.protocols:
            folds = build_folds(protocol, prot=prot, tcfg=tcfg, year=year,
                                scaffold=scaffold, n=n)
            if not folds:
                print(f"[e04g] {slug}/{protocol} valid fold skip", flush=True)
                append_rows(skip_path, [{"target": slug, "reason": f"{protocol}: valid fold"}])
                continue
            print(f"[e04g] {slug}/{protocol}: {len(folds)} folds", flush=True)

            for cname, mname, params in configs:
                rows: list[dict] = []
                prows: list[dict] = []
                t0 = time.time()
                for fold_name, tr, te in folds:
                    for seed in args.seeds:
                        
                        
                        # torch.manual_seed(self.seed) / pl.seed_everything(self.seed)，
                        
                        
                        if only is not None and\
                                (slug, protocol, cname, fold_name, seed) not in only:
                            continue
                        try:
                            mdl = build_gnn(mname, params, seed=seed, device=args.device)
                            mdl.fit(smiles[tr], y[tr])
                            pred = np.asarray(mdl.predict(smiles[te]), dtype=float)
                        except Exception as exc:                    # noqa: BLE001
                            print(f"[e04g] !! {slug}/{protocol}/{cname}/{fold_name}/s{seed} "
                                  f"failed: {type(exc).__name__}: {exc}", flush=True)
                            append_rows(skip_path, [{
                                "target": slug,
                                "reason": f"{protocol}/{cname}/{fold_name}/s{seed}: "
                                          f"{type(exc).__name__}: {exc}"}])
                            continue
                        ok = np.isfinite(pred)
                        
                        met = compute_metrics(y[te][ok], pred[ok], enrichment_pct=enrich)
                        rows.append({
                            "target": slug, "protocol": protocol, "fold": fold_name,
                            "model": cname, "algo": mname, "family": GNN_FAMILY,
                            "seed": seed, "n_train": int(len(tr)),
                            "n_test": int(ok.sum()), "n_pred_nan": int((~ok).sum()),
                            **met,
                        })
                        
                        
                        for j, ti in enumerate(te):
                            prows.append({
                                "target": slug, "protocol": protocol, "fold": fold_name,
                                "model": cname, "seed": seed, "mol_id": str(mol_id[ti]),
                                "y_true": float(y[ti]), "y_pred": float(pred[j]),
                            })
                append_rows(pf_path, rows)
                append_rows(pred_path, prows)
                print(f"[e04g] {cname:18s} {len(rows):3d} {time.time() - t0:6.1f}s",
                      flush=True)

        write_run_meta(outdir, target=slug, stage=args.out_name,
                       protocol_fingerprint=prot.fingerprint,
                       extra={
                           "script": "scripts/e04g_hparam_pool_gnn.py",
                           "purpose": "GNN e04 note_8/43",
                           "declaration": "manuscript data declaration",
                           "grid_file": str(GRID_PATH.relative_to(ROOT))
                                        ,
                           "k": args.k, "sample_seed": prot.seed,
                           "configs": [c[0] for c in configs],
                           "protocols": args.protocols, "seeds": args.seeds,
                           "early_stopping": False,
                           "aggregation": "median",
                           "collapsed_configs": "resultsresults",
                           "not_done": ["pooled", "molecules bootstrap"],
                       })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
