#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits                    # noqa: E402

apply_thread_limits(1)

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from pparbench.config import RESULTS_DIR, load_protocol, load_target  # noqa: E402
from pparbench.io import features_path                                # noqa: E402
from r2_gnn_matrix import PROTOCOL_ORDER, build_folds                 # noqa: E402


def first_round_dir(slug: str) -> Path:
    return RESULTS_DIR / slug / "p03_protocol_matrix"


def check_target(slug: str) -> list[str]:
    """Documentation."""
    problems: list[str] = []
    d1 = first_round_dir(slug)
    pred_file = d1 / "predictions.csv"
    if not pred_file.exists():
        return [f"{slug}: missing {pred_file} target？"]

    prot = load_protocol()
    tcfg = load_target(slug)
    npz = np.load(features_path(slug, "ligand2057"), allow_pickle=True)
    y, year, scaffold = npz["y"], npz["year"], npz["scaffold"]
    n = len(y)

    pred = pd.read_csv(pred_file)
    for protocol in PROTOCOL_ORDER:
        got = build_folds(protocol, prot=prot, tcfg=tcfg, year=year,
                          scaffold=scaffold, n=n)
        ref = pred[pred.protocol == protocol]
        if ref.empty:
            if got:
                problems.append(f"{slug}/{protocol}: {len(got)} folds")
            continue
        
        one = ref[ref.model == ref.model.iloc[0]]
        ref_folds = {f: set(g.test_idx.astype(int)) for f, g in one.groupby("fold")}
        got_folds = {name: set(int(i) for i in te) for name, te, in
                     [(nm, te) for nm, _tr, te in got]}

        if set(ref_folds) != set(got_folds):
            problems.append(
                f"{slug}/{protocol}: fold inconsistent"
                f"={sorted(ref_folds)} ={sorted(got_folds)}")
            continue
        for fname in sorted(ref_folds):
            if ref_folds[fname] != got_folds[fname]:
                a, b = ref_folds[fname], got_folds[fname]
                problems.append(
                    f"{slug}/{protocol}/{fname}: test_idx"
                    f"( n={len(a)}, n={len(b)},"
                    f"{len(a - b)}, {len(b - a)})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="/")
    ap.add_argument("--targets", nargs="*", default=["all"])
    args = ap.parse_args()

    if args.targets == ["all"]:
        
        man = pd.read_csv(ROOT / "configs" / "manifests" / "targets_manifest.csv", comment="#")
        targets = sorted(man["slug"].astype(str).tolist())
    else:
        targets = list(args.targets)

    all_problems: list[str] = []
    for slug in targets:
        try:
            probs = check_target(slug)
        except Exception as exc:                                     # noqa: BLE001
            probs = [f"{slug}: failed {type(exc).__name__}: {exc}"]
        status = "OK  " if not probs else "FAIL"
        print(f"[{status}] {slug}")
        for p in probs:
            print(f"        {p}")
        all_problems += probs

    print()
    if all_problems:
        print(f"✗ {len(all_problems)} inconsistent mixed7 **** trees4")
        return 1
    print(f"✓ {len(targets)} targetconsistent mixed7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
