#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()          

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402
from scipy.stats import spearmanr                      # noqa: E402
from scipy.stats import binomtest                      # noqa: E402

from pparbench.io import save_table, write_run_meta     # noqa: E402
from e17_r2_axis import FROZEN20                        

PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]
PAIRS = [("random_holdout", "scaffold_cv"),
         ("scaffold_cv", "temporal"),
         ("random_holdout", "temporal")]

TOL = 5e-4          
TIE_EPS = 1e-3      



def assert_scope(manifest: Path) -> list[str]:
    """Documentation."""
    slugs = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head = line.split(",")[0].strip()
        if head and head != "slug":
            slugs.append(head)
    expect = set(slugs) | {"pparg"}
    if set(FROZEN20) != expect:
        raise SystemExit(
            f"⛔ inconsistent 159/50 \n"
            f"FROZEN20 {len(FROZEN20)} \n"
            f"manifest 19 ∪ {{pparg}} {len(expect)} \n"
            f"FROZEN20 {sorted(set(FROZEN20) - expect)}\n"
            f"manifest {sorted(expect - set(FROZEN20))}")
    print(f"[] FROZEN20 == manifest {len(slugs)} ∪ {{pparg}} ✅ pparg curated_csv manifest")
    return sorted(FROZEN20)



def foldmean_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    out = []
    for (proto, model), grp in pred.groupby(["protocol", "model"], sort=False):
        rhos, collisions = [], 0
        for _, fold in grp.groupby("fold", sort=False):
            yt = fold["y_true"].to_numpy(dtype=float)
            yp = fold["y_pred"].to_numpy(dtype=float)
            if len(yt) < 3:
                continue
            rhos.append(spearmanr(yt, yp).statistic)
            uniq, counts = np.unique(yp, return_counts=True)
            collisions += int((counts * (counts - 1) // 2).sum())
        if not rhos:
            continue
        out.append({"protocol": proto, "model": model,
                    "spearman_foldmean": float(np.mean(rhos)),
                    "n_folds": len(rhos), "n_pred_collisions": collisions})
    return pd.DataFrame(out)


def winners_for_target(slug: str, results: Path) -> tuple[dict, pd.DataFrame, list[str]]:
    """Documentation."""
    tdir = results / slug / "p03_protocol_matrix"
    pred = pd.read_csv(tdir / "predictions.csv")
    rank = pd.read_csv(tdir / "ranking.csv")
    rank = rank[rank["aggregation"] == "fold_mean"]

    fm = foldmean_from_predictions(pred)
    fam = (pd.read_csv(tdir / "aggregated.csv")[["model", "family"]]
           .drop_duplicates().set_index("model")["family"].to_dict())
    fm["family"] = fm["model"].map(fam)
    fm.insert(0, "target", slug)

    
    problems = []
    merged = fm.merge(rank[["protocol", "model", "spearman"]], on=["protocol", "model"],
                      how="inner", suffixes=("", "_frozen"))
    dev = (merged["spearman_foldmean"] - merged["spearman"]).abs()
    if len(merged) == 0:
        problems.append(f"{slug}: ranking.csv predictions.csv (protocol, model)")
    elif dev.max() > TOL:
        bad = merged.loc[dev.idxmax()]
        problems.append(f"{slug}: {dev.max():.2e} > {TOL:.0e}"
                        f"（{bad['protocol']}/{bad['model']}）")

    win = {}
    for proto in PROTOCOLS:
        sub = fm[fm["protocol"] == proto].sort_values("spearman_foldmean", ascending=False)
        if sub.empty:
            problems.append(f"{slug}/{proto}: predictiondata")
            continue
        top = sub.iloc[0]
        margin = float(top["spearman_foldmean"] - sub.iloc[1]["spearman_foldmean"]) if len(sub) > 1 else np.nan
        win[proto] = {"model": top["model"], "family": top["family"],
                      "spearman": float(top["spearman_foldmean"]),
                      "margin": margin,
                      "runner_up": sub.iloc[1]["model"] if len(sub) > 1 else "",
                      "runner_up_family": sub.iloc[1]["family"] if len(sub) > 1 else "",
                      "n_pred_collisions": int(top["n_pred_collisions"])}
    return win, fm, problems



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "manifests" / "targets_manifest.csv"))
    ap.add_argument("--tag", default="", help="output")
    args = ap.parse_args()

    results = Path(args.results)
    slugs = assert_scope(Path(args.manifest))

    rows, foldmeans, problems = [], [], []
    for slug in slugs:
        win, fm, prob = winners_for_target(slug, results)
        problems += prob
        foldmeans.append(fm)
        row = {"target": slug}
        for proto in PROTOCOLS:
            w = win.get(proto, {})
            row[proto] = w.get("model", "")
            row[f"{proto}_fam"] = w.get("family", "")
            row[f"{proto}_rho"] = w.get("spearman", np.nan)
            row[f"{proto}_margin"] = w.get("margin", np.nan)
            row[f"{proto}_runner_up"] = w.get("runner_up", "")
            row[f"{proto}_runner_up_fam"] = w.get("runner_up_family", "")
            row[f"{proto}_collisions"] = w.get("n_pred_collisions", -1)
        rows.append(row)
        print(f"[{slug:9s}] " + "  ".join(
            f"{p[:4]}={win.get(p, {}).get('model', '?'):11s}(Δ{win.get(p, {}).get('margin', float('nan')):.5f})"
            for p in PROTOCOLS))

    if problems:
        raise SystemExit("⛔ failed \n" + "\n  ".join(problems))
    print("[] ranking.csv %.4f 5e-4 ✅")

    win_df = pd.DataFrame(rows)

    
    shifts = []
    for a, b in PAIRS:
        chg_model = (win_df[a] != win_df[b])
        chg_fam = (win_df[f"{a}_fam"] != win_df[f"{b}_fam"])
        shifts.append({"from": a, "to": b, "n_targets": len(win_df),
                       "changed_model": int(chg_model.sum()),
                       "changed_family": int(chg_fam.sum()),
                       "changed_model_targets": ";".join(win_df.loc[chg_model, "target"]),
                       "changed_family_targets": ";".join(win_df.loc[chg_fam, "target"])})
    shift_df = pd.DataFrame(shifts)

    
    fams = sorted({f for p in PROTOCOLS for f in win_df[f"{p}_fam"] if f})
    fam_df = pd.DataFrame([
        {"protocol": p, **{f: int((win_df[f"{p}_fam"] == f).sum()) for f in fams}}
        for p in PROTOCOLS])

    
    
    mc = []
    for what, cols in [("changed_model", (PROTOCOLS[0], PROTOCOLS[1], PROTOCOLS[2])),
                       ("changed_family", (PROTOCOLS[0], PROTOCOLS[1], PROTOCOLS[2]))]:
        key = "" if what == "changed_model" else "_fam"
        scaf = win_df[f"{cols[0]}{key}"] != win_df[f"{cols[1]}{key}"]     
        temp = win_df[f"{cols[1]}{key}"] != win_df[f"{cols[2]}{key}"]     
        b = int((~scaf & temp).sum())      
        c = int((scaf & ~temp).sum())      
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else np.nan
        mc.append({"what": what, "only_temporal": b, "only_scaffold": c,
                   "both": int((scaf & temp).sum()), "neither": int((~scaf & ~temp).sum()),
                   "p_exact_mcnemar": p})
    mc_df = pd.DataFrame(mc)

    
    tight = []
    for _, r in win_df.iterrows():
        for p in PROTOCOLS:
            if pd.notna(r[f"{p}_margin"]) and r[f"{p}_margin"] < TIE_EPS:
                tight.append({"target": r["target"], "protocol": p, "winner": r[p],
                              "runner_up": r[f"{p}_runner_up"], "margin": r[f"{p}_margin"],
                              
                              
                              "winner_family": r[f"{p}_fam"],
                              "runner_up_family": r[f"{p}_runner_up_fam"],
                              "cross_family": r[f"{p}_fam"] != r[f"{p}_runner_up_fam"]})
    tight_df = pd.DataFrame(tight)

    
    
    
    
    
    sens = []
    if len(tight_df):
        flippable = [(r["target"], r["protocol"], r["runner_up"], r["runner_up_family"])
                     for _, r in tight_df.iterrows()]
        for a, b in PAIRS:
            counts_model, counts_fam = set(), set()
            for combo in itertools.product([False, True], repeat=len(flippable)):
                w = win_df.copy()
                for flip, (tgt, proto, ru, ruf) in zip(combo, flippable):
                    if flip:
                        i = w.index[w["target"] == tgt][0]
                        w.at[i, proto] = ru
                        w.at[i, f"{proto}_fam"] = ruf
                counts_model.add(int((w[a] != w[b]).sum()))
                counts_fam.add(int((w[f"{a}_fam"] != w[f"{b}_fam"]).sum()))
            sens.append({"from": a, "to": b,
                         "changed_model_point": int((win_df[a] != win_df[b]).sum()),
                         "changed_model_min": min(counts_model), "changed_model_max": max(counts_model),
                         "changed_family_point": int((win_df[f"{a}_fam"] != win_df[f"{b}_fam"]).sum()),
                         "changed_family_min": min(counts_fam), "changed_family_max": max(counts_fam),
                         "n_flippable_cells": len(flippable), "tie_eps": TIE_EPS})
    sens_df = pd.DataFrame(sens)

    outdir = Path(args.results) / "cross_target" / ("e15_winner_shift" + (f"__{args.tag}" if args.tag else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in [("e15_winners", win_df), ("e15_shift_counts", shift_df),
                     ("e15_family_distribution", fam_df), ("e15_mcnemar", mc_df),
                     ("e15_tight_margins", tight_df), ("e15_tie_sensitivity", sens_df),
                     ("e15_foldmean_fullprec", pd.concat(foldmeans, ignore_index=True))]:
        save_table(df, outdir / f"{name}.csv", float_fmt="%.8f")   

    print("\n=== ===")
    print(shift_df[["from", "to", "changed_model", "changed_family"]].to_string(index=False))
    print("\n=== ===")
    print(fam_df.to_string(index=False))
    print("\n=== McNemar ===")
    print(mc_df.to_string(index=False))
    print(f"\n=== < {TIE_EPS} ===")
    print(tight_df.to_string(index=False) if len(tight_df) else "message")
    print(f"\n=== ===")
    print(sens_df.to_string(index=False) if len(sens_df) else "message")

    write_run_meta(outdir, target="cross_target", stage="e15_winner_shift",
                   protocol_fingerprint="n/a extra", extra={
        "what": "20 target note_8/30 §2.3.1 3/20 vs 14/20 /tmp",
        "scope": f"FROZEN20 == manifest 19 ∪ {{pparg}} n={len(slugs)}",
        "aggregation": "fold_mean pooled",
        "winner_source": ("predictions.csv fold-mean spearman argmax"
                          "**** ranking.csv rank/spearman %.4f 114/135"),
        "self_check": f"ranking.csv %.4f {TOL}",
        "known_tie_at_4dp": ("esr1/random_holdout %.4f lightgbm extratrees 0.8849"
                             "lightgbm e15_tight_margins.csv"),
        "not_done": [
            "predictions.csv y_pred %.4f 1e-4"
            "n_pred_collisions",
            "target no 163"
            "McNemar",
        ],
        "red_lines": [25, 29, 46, 50, 114, 135, 142, 159, 163, 18],
    })


if __name__ == "__main__":
    main()
