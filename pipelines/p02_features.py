from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pparbench.config import load_protocol, load_target
from pparbench.features import (build_ad_fingerprints, build_ligand_features,
                                murcko_scaffolds)
from pparbench.io import (canonical_dataset_path, features_path,
                          target_data_dir, target_results_dir, write_run_meta)


def main() -> int:
    ap = argparse.ArgumentParser(description="features")
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    prot = load_protocol()
    tcfg = load_target(args.target)

    ds_path = canonical_dataset_path(tcfg.slug)
    if not ds_path.exists():
        print(f"[error] missing {ds_path} run first p01_curate.py --target {tcfg.slug}")
        return 1

    df = pd.read_csv(ds_path)
    smiles = df["smiles"].astype(str).tolist()
    print(f"[p02] target={tcfg.slug}  n={len(df)}")

    X, ok, names = build_ligand_features(smiles, prot.features)
    n_bad = int((~ok).sum())
    if n_bad:
        
        print(f"[warn] {n_bad} moleculesfeaturesfailed")
    print(f"[p02] X shape = {X.shape} ( 2057 )")

    scaffolds = murcko_scaffolds(smiles)
    n_scaf = len(set(s for s in scaffolds if s))
    print(f"[p02] Murcko scaffold = {n_scaf} (scaffold/molecules = {n_scaf/len(df):.2f})")

    np.savez_compressed(
        features_path(tcfg.slug, "ligand2057"),
        X=X.astype(np.float32),
        y=df["y"].to_numpy(dtype=np.float32),
        year=df["year"].to_numpy(dtype=np.float64),
        mol_id=np.array(df["mol_id"], dtype=object),
        scaffold=np.array(scaffolds, dtype=object),
        ok=ok,
        feature_names=np.array(names, dtype=object),
    )
    print(f"[p02] write {features_path(tcfg.slug, 'ligand2057')}")

    
    ad_fps = build_ad_fingerprints(smiles, prot.applicability_domain)
    ad_bits = np.zeros((len(smiles), int(prot.applicability_domain["fp_bits"])),
                       dtype=np.uint8)
    from rdkit import DataStructs
    for i, fp in enumerate(ad_fps):
        if fp is None:
            continue
        arr = np.zeros((ad_bits.shape[1],), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        ad_bits[i] = arr.astype(np.uint8)
    np.savez_compressed(target_data_dir(tcfg.slug) / "features_ad.npz",
                        bits=ad_bits, mol_id=np.array(df["mol_id"], dtype=object))
    print(f"[p02] write {target_data_dir(tcfg.slug) / 'features_ad.npz'}")

    write_run_meta(
        target_results_dir(tcfg.slug, "p02_features"),
        target=tcfg.slug, stage="p02_features",
        protocol_fingerprint=prot.fingerprint,
        extra={"n_molecules": len(df), "n_features": int(X.shape[1]),
               "n_featurization_failed": n_bad, "n_scaffolds": n_scaf},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
