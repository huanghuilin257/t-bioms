from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pparbench.config import list_targets, load_protocol, load_target
from pparbench.curate import build_dataset
from pparbench.io import (canonical_dataset_path, save_table, target_data_dir,
                          target_results_dir, write_run_meta)
from pparbench.splits import suggest_cutpoints


def main() -> int:
    ap = argparse.ArgumentParser(description="single targetdata")
    ap.add_argument("--target", required=True, help=f"target slug available: {list_targets()}")
    ap.add_argument("--force", action="store_true", help="dataset.csv")
    args = ap.parse_args()

    prot = load_protocol()
    tcfg = load_target(args.target)
    out_csv = canonical_dataset_path(tcfg.slug)

    if out_csv.exists() and not args.force:
        print(f"[skip] {out_csv} --force")
        return 0

    print(f"[p01] target={tcfg.slug} ({tcfg.name})  source={tcfg.source}  endpoint={tcfg.endpoint}")
    df = build_dataset(tcfg)

    log = df.attrs.get("finalize_log", {})
    print(f"[p01] molecules={len(df)} invalidSMILES={log.get('invalid_smiles_dropped')}"
          f"duplicate structures merged={log.get('duplicate_structures_merged')}")
    print(f"[p01] year {log.get('year_min')}–{log.get('year_max')}"
          f"missing year={log.get('n_missing_year')}")

    
    frac_missing = float(df["year"].isna().mean())
    if frac_missing > 0.05:
        print(f"[warn] {frac_missing:.1%} moleculesmissing year temporal"
              f"paths.chembl_export molecules")

    save_table(df, out_csv)
    print(f"[p01] write {out_csv}")

    if "curation_log" in df.attrs:
        p = save_table(df.attrs["curation_log"], target_data_dir(tcfg.slug) / "curation_log.csv",
                       float_fmt="%.0f")
        print(f"[p01] write {p}")

    
    years = df["year"].to_numpy(dtype=float)
    yr_counts = (df["year"].value_counts().sort_index()
                 .rename_axis("year").reset_index(name="n_molecules"))
    yr_counts["cumulative"] = yr_counts["n_molecules"].cumsum()
    save_table(yr_counts, target_data_dir(tcfg.slug) / "dataset_summary.csv", float_fmt="%.0f")

    suggested = suggest_cutpoints(years, min_test=prot.protocols["temporal"]["min_test"])
    print(f"[p01] suggested cutpoints yaml cutpoints : {suggested}")
    print(f"[p01] current YAML cutpoints = {tcfg.cutpoints}")
    if tcfg.cutpoints:
        rows = []
        for T in tcfg.cutpoints:
            rows.append({"cutpoint": T,
                         "n_train": int(np.sum(years < T)),
                         "n_test": int(np.sum(years >= T))})
        print(pd.DataFrame(rows).to_string(index=False))

    write_run_meta(
        target_results_dir(tcfg.slug, "p01_curate"),
        target=tcfg.slug, stage="p01_curate",
        protocol_fingerprint=prot.fingerprint,
        extra={"source": tcfg.source, "finalize_log": log,
               "suggested_cutpoints": suggested, "yaml_cutpoints": tcfg.cutpoints},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
