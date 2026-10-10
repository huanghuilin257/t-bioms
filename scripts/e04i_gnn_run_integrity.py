#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pparbench.io import save_table, write_run_meta  # noqa: E402

OUTDIR = ROOT / "results" / "cross_target" / "e04i_gnn_run_integrity"


EXPECTED = {"events": 1380, "row_keys": 1371, "cells": 93, "targets": 12}


REASON_RE = re.compile(r"^([a-z_]+)/([a-z]+#\d+)/([^/]+)/s(\d+):")




ALLOC_RE = re.compile(r"Tried to allocate ([\d.]+) (MiB|GiB|KiB)")
CAP_RE = re.compile(r"total capacity of ([\d.]+) (GiB|MiB)")
ALLOC_SHARE_CLAIM = 0.02  


def _to_mib(v: str, unit: str) -> float:
    x = float(v)
    return x * 1024 if unit == "GiB" else (x / 1024 if unit == "KiB" else x)


def check_alloc_share(reasons) -> dict:
    """Documentation."""
    shares = []
    n_bad = 0
    for r in reasons:
        m1, m2 = ALLOC_RE.search(r), CAP_RE.search(r)
        if not (m1 and m2):
            n_bad += 1
            continue
        shares.append(_to_mib(*m1.groups()) / _to_mib(*m2.groups()))
    if n_bad:
        raise SystemExit(f"⛔ {n_bad} ⇒ 2%")
    mx = max(shares)
    return {
        "n_events_checked": len(shares),
        "max_share_of_device": round(mx, 6),
        "n_at_or_above_claim": int(sum(1 for x in shares if x >= ALLOC_SHARE_CLAIM)),
        "claim_bound": ALLOC_SHARE_CLAIM,
        "claim_holds": bool(mx < ALLOC_SHARE_CLAIM),
        "n_at_or_above_1pct": int(sum(1 for x in shares if x >= 0.01)),
    }


def load_skipped(merged_name: str) -> pd.DataFrame:
    """Documentation."""
    frames = []
    for base in ("main_skipped.csv", "topup_skipped.csv"):
        for f in sorted(glob.glob(str(ROOT / "results" / "*" / merged_name / base))):
            d = pd.read_csv(f)
            if d.empty:
                continue
            d["source_file"] = base
            frames.append(d)
    if not frames:
        raise SystemExit(f"⛔ skip results/*/{merged_name}/")
    a = pd.concat(frames, ignore_index=True)

    n_raw = len(a)
    
    ev = a.drop_duplicates(subset=["target", "reason"]).reset_index(drop=True)

    parsed = ev["reason"].apply(
        lambda r: pd.Series(
            REASON_RE.match(r).groups() if REASON_RE.match(r) else (None, None, None, None)
        )
    )
    parsed.columns = ["protocol", "model", "fold", "seed"]
    ev = pd.concat([ev[["target", "source_file"]], parsed], axis=1)
    n_bad = int(ev["protocol"].isna().sum())
    if n_bad:
        raise SystemExit(f"⛔ {n_bad} reason ⇒")
    ev["algo"] = ev["model"].str.split("#").str[0]
    ev.attrs["n_raw"] = n_raw
    return ev


def load_per_fold(merged_name: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / merged_name / "per_fold.csv"))):
        frames.append(pd.read_csv(f, float_precision="round_trip"))
    if not frames:
        raise SystemExit(f"⛔ per_fold.csv results/*/{merged_name}/")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-name", default="e04g_hparam_pool_gnn_merged",
                    help="target merged default canonical merged")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    
    ev = load_skipped(args.merged_name)
    keycols = ["target", "protocol", "model", "fold", "seed"]
    keys = ev.drop_duplicates(subset=keycols)

    observed = {
        "events": len(ev),
        "row_keys": len(keys),
        "cells": len(keys.drop_duplicates(subset=["target", "protocol", "fold"])),
        "targets": int(keys["target"].nunique()),
    }
    print("[e04i] skip note_8/00:670")
    for k, want in EXPECTED.items():
        got = observed[k]
        mark = "✅" if got == want else "⛔"
        print(f"{mark} {k:9s} {want:6d} {got:6d}")
    bad = [k for k in EXPECTED if observed[k] != EXPECTED[k]]
    if bad:
        raise SystemExit(f"⛔ failed {bad} ⇒ ⛔")

    algos = sorted(ev["algo"].unique())
    print(f"[e04i] {algos} ⛔ dmpnn")

    
    raw_reasons = []
    for base in ("main_skipped.csv", "topup_skipped.csv"):
        for f in sorted(glob.glob(str(ROOT / "results" / "*" / args.merged_name / base))):
            d = pd.read_csv(f)
            if len(d):
                raw_reasons.append(d)
    reasons = (pd.concat(raw_reasons, ignore_index=True)
               .drop_duplicates(subset=["target", "reason"])["reason"])
    alloc = check_alloc_share(reasons)
    mark = "✅" if alloc["claim_holds"] else "⛔"
    print(f"[e04i] {mark} failed"
          f"{ALLOC_SHARE_CLAIM:.0%} {alloc['max_share_of_device']:.4%}"
          f"≥ {alloc['n_at_or_above_claim']} / {alloc['n_events_checked']}")
    print(f"[e04i] ⚠ ≥ 1% {alloc['n_at_or_above_1pct']}"
          f"⇒ ⛔ 1%")

    per_t = (
        keys.groupby("target")
        .agg(
            row_keys=("model", "size"),
            cells=("fold", lambda s: 0),  
        )
        .reset_index()
    )
    cell_cnt = (
        keys.drop_duplicates(subset=["target", "protocol", "fold"])
        .groupby("target").size().rename("cells").reset_index()
    )
    ev_cnt = ev.groupby("target").size().rename("events").reset_index()
    per_t = per_t.drop(columns=["cells"]).merge(cell_cnt, on="target").merge(ev_cnt, on="target")
    
    prot = (
        keys.groupby(["target", "protocol"]).size().unstack(fill_value=0)
        .add_prefix("row_keys_").reset_index()
    )
    per_t = per_t.merge(prot, on="target", how="left").fillna(0)
    for c in per_t.columns:
        if c != "target":
            per_t[c] = per_t[c].astype(int)
    per_t = per_t.sort_values("row_keys", ascending=False).reset_index(drop=True)
    print("\n[e04i] skip")
    print(per_t.to_string(index=False))

    
    pf = load_per_fold(args.merged_name)
    n_rows_total = len(pf)
    degen = pf[(pf["pred_std"] == 0) & (pf["n_pred_nan"] == 0)]
    print(f"\n[e04i] per_fold {n_rows_total} prediction {len(degen)}")
    print(f"[e04i] prediction {sorted(degen['algo'].unique())}")

    deg_t = (
        degen.groupby("target")
        .agg(constant_rows=("pred_std", "size"),
             cells=("fold", lambda s: 0))
        .reset_index().drop(columns=["cells"])
    )
    deg_cells = (
        degen.drop_duplicates(subset=["target", "protocol", "fold"])
        .groupby("target").size().rename("cells_touched").reset_index()
    )
    deg_t = deg_t.merge(deg_cells, on="target")
    
    tot_t = pf.groupby("target").size().rename("rows_total").reset_index()
    deg_t = deg_t.merge(tot_t, on="target").sort_values("constant_rows", ascending=False).reset_index(drop=True)
    print("\n[e04i] prediction")
    print(deg_t.to_string(index=False))

    
    by_cfg = (
        degen.groupby(["model", "protocol"]).size().unstack(fill_value=0)
        .add_prefix("rows_").reset_index()
    )
    by_cfg["rows_total"] = by_cfg[[c for c in by_cfg.columns if c.startswith("rows_")]].sum(axis=1)
    all_cfgs = sorted(pf.loc[pf["algo"].isin(degen["algo"].unique()), "model"].unique())
    missing = [c for c in all_cfgs if c not in set(by_cfg["model"])]
    by_cfg = pd.concat(
        [by_cfg, pd.DataFrame([{"model": c} for c in missing])], ignore_index=True
    ).fillna(0)
    for c in by_cfg.columns:
        if c != "model":
            by_cfg[c] = by_cfg[c].astype(int)
    by_cfg = by_cfg.sort_values("rows_total", ascending=False).reset_index(drop=True)
    print("\n[e04i] configuration prediction ⛔ 0 configuration ⛔")
    print(by_cfg.to_string(index=False))

    
    grp = degen.groupby(["target", "protocol", "fold", "model"]).size().rename("n_seeds_constant")
    seed_pat = grp.value_counts().sort_index().rename("n_config_cell_pairs").reset_index()
    seed_pat.columns = ["n_seeds_constant", "n_config_cell_pairs"]
    n_seeds_per_cell = int(pf.groupby(["target", "protocol", "fold", "model"]).size().max())
    seed_pat["all_seeds"] = n_seeds_per_cell
    print(f"\n[e04i] (configuration, ) {len(grp)} {n_seeds_per_cell} seed")
    print(seed_pat.to_string(index=False))
    n_full = int((grp == n_seeds_per_cell).sum())
    print(f"[e04i] seed {n_full} seed {len(grp) - n_full}"
          f"⇒ ⛔ output****configuration")

    
    s_oom = set(per_t["target"])
    s_deg = set(deg_t["target"])
    print(f"\n[e04i] OOM {len(s_oom)} prediction {len(s_deg)} {len(s_oom & s_deg)}")

    
    
    for name, df in (("e04i_per_target_oom", per_t),
                     ("e04i_per_target_degenerate", deg_t),
                     ("e04i_degenerate_by_config", by_cfg),
                     ("e04i_degenerate_seed_pattern", seed_pat)):
        save_table(df, outdir / f"{name}.csv")
        df.to_csv(outdir / f"{name}__fullprec.csv", index=False)

    write_run_meta(
        outdir, target="cross_target", stage="e04i_gnn_run_integrity",
        protocol_fingerprint=f"merged={args.merged_name};degenerate=pred_std==0&n_pred_nan==0",
        extra={
            "tool": "scripts/e04i_gnn_run_integrity.py",
            "inputs": [
                f"results/*/{args.merged_name}/main_skipped.csv (read-only)",
                f"results/*/{args.merged_name}/topup_skipped.csv (read-only)",
                f"results/*/{args.merged_name}/per_fold.csv (read-only)",
            ],
            "purpose": "S13 + output",
            "scope_rule": (
                "= merged skip ⛔"
                "159 note_8/37 M4.2"
            ),
            "reproduced_totals": observed,
            "expected_totals": EXPECTED,
            "units": {
                "row_key": "(target, protocol, model, fold, seed) skipped.csv",
                "cell": "(target, protocol, fold) 72",
                "per_fold_row": "(target, protocol, fold, model, seed) output",
            },
            "degenerate_criterion": "pred_std == 0 and n_pred_nan == 0",
            "oom_algos_observed": algos,
            "degenerate_algos_observed": sorted(degen["algo"].unique()),
            "degenerate_rows": int(len(degen)),
            "rows_per_algo": {k: int(v) for k, v in pf.groupby("algo").size().items()},
            "degenerate_cells_touched": int(len(degen.drop_duplicates(
                subset=["target", "protocol", "fold"]))),
            "degenerate_targets": int(degen["target"].nunique()),
            "degenerate_config_cell_pairs": int(len(grp)),
            "degenerate_all_seeds_pairs": n_full,
            "degenerate_partial_seeds_pairs": int(len(grp)) - n_full,
            "configs_with_zero_degenerate": missing,
            "alloc_share_claim_check": alloc,
            "no_attribution": (
                "⛔ 175/149 ⛔ OOM output"
                "model"
            ),
            "not_here": (
                "worker / reason MiB"
                "16 × §5.2.26 (2) ⛔ ⇒"
            ),
        },
    )
    print(f"\n[e04i] write {outdir}")


if __name__ == "__main__":
    main()
