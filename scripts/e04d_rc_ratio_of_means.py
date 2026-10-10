#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits  # noqa: E402

apply_thread_limits()

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import e04_level_dissociation as e04ld                                       # noqa: E402
from e04_level_dissociation import (CLUSTERS, GNN_ALGOS, TREE_FAMILIES,     # noqa: E402
                                    analyse_target, build_agg, e04g_complete,
                                    widest_ci)
from e04_random_floor import ratio_of_means_ci  # noqa: E402
from e17_r2_axis import FROZEN20              # noqa: E402
from pparbench.io import save_table           # noqa: E402

E04LD = ROOT / "results" / "cross_target" / "e04_level_dissociation"

FROZEN = E04LD / "trees_only_4algo_2family"
OUT = ROOT / "results" / "cross_target" / "e04d_rc_ratio_of_means"
KEY = ["metric", "selector", "level", "k"]

TOL_WRITE = 1e-6


def build_scope(with_l3: bool, exclude_gin: bool, gnn_agg: str) -> tuple[dict, list, str]:
    """Documentation."""
    families = dict(TREE_FAMILIES)
    if with_l3:
        families["gnn"] = [a for a in GNN_ALGOS if not (exclude_gin and a == "gin")]
    algos = [a for fam in families.values() for a in fam]
    scope = ("trees_only_4algo_2family" if not with_l3
             else f"full_{len(algos)}algo_{len(families)}family"
                  f"{'_no_gin' if exclude_gin else '_with_gin'}__{gnn_agg}")
    return families, algos, scope


def recompute(families: dict, algos: list, *, with_l3: bool,
              exclude_gin: bool, gnn_agg: str) -> pd.DataFrame:
    rows = []
    for slug in FROZEN20:
        agg, diag = build_agg(slug, include_gnn=with_l3, gnn_agg=gnn_agg,
                              exclude_gin=exclude_gin)
        if diag["n_nan_unexplained"]:
            raise SystemExit(f"⛔ {slug}: NaN")
        rows += analyse_target(slug, agg, algos=algos, families=families,
                              with_l3=with_l3)
    return pd.DataFrame(rows)


def check_reproduces(pt: pd.DataFrame, frozen: Path) -> None:
    """Documentation."""
    fz = pd.read_csv(frozen / "per_target_level.csv")
    cols = ["target"] + KEY
    m = pt.merge(fz, on=cols, suffixes=("_new", "_fz"))
    if len(m) != len(fz):
        raise SystemExit(f"⛔ ① {len(pt)} / {len(fz)} / {len(m)}")
    worst = {}
    for c in ("mean_regret", "mean_floor", "mean_spread", "regret_over_floor",
              "frac_zero_regret", "mean_n_ties", "frac_tied"):
        a = m[f"{c}_new"].to_numpy(float); b = m[f"{c}_fz"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any():
            raise SystemExit(f"⛔ ① {c} NaN inconsistent")
        d = float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0
        worst[c] = d
        if d > TOL_WRITE:
            raise SystemExit(f"⛔ ① {c} {d:.3e} > {TOL_WRITE:.0e}"
                             f"⇒")
    print(f"[e04d] ✅ ① {len(m)}"
          f"{max(worst.values()):.2e} {max(worst, key=worst.get)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="R-C regret_over_floor")
    ap.add_argument("--allow-l3", action="store_true",
                    help="L3 e04g 20/20")
    ap.add_argument("--exclude-gin", action="store_true",
                    help="123 gin --allow-l3")
    ap.add_argument("--gnn-agg", default="seed_median",
                    choices=["seed_median", "fold_mean_only"])
    ap.add_argument("--e04g-dir", default="e04g_hparam_pool_gnn_merged",
                    help="GNN default")
    args = ap.parse_args()
    if not args.allow_l3 and (args.exclude_gin or args.gnn_agg != "seed_median"):
        raise SystemExit("⛔ --exclude-gin / --gnn-agg --allow-l3")

    with_l3 = False
    if args.allow_l3:
        e04ld.E04G_DIR = args.e04g_dir          
        ok, missing, _ = e04g_complete(verbose=True)
        if not ok:
            print(f"⛔ L3 e04g {len(missing)}/20 -> {missing}")
            return 2
        with_l3 = True

    families, algos, scope = build_scope(with_l3, args.exclude_gin, args.gnn_agg)
    frozen = E04LD / scope
    if not (frozen / "per_target_level.csv").exists():
        raise SystemExit(f"⛔ {frozen} ⇒ run first e04_level_dissociation")
    
    out = OUT if not with_l3 else OUT / scope
    print(f"[e04d] = {scope} algos = {algos} = {frozen.name}")

    pt = recompute(families, algos, with_l3=with_l3,
                   exclude_gin=args.exclude_gin, gnn_agg=args.gnn_agg)
    check_reproduces(pt, frozen)
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")

    fz_sum = pd.read_csv(frozen / "level_summary.csv")
    fz_rof = fz_sum[fz_sum.quantity == "regret_over_floor"].set_index(KEY)

    rows = []
    for keys, g in pt.groupby(KEY):
        g = g.sort_values("target")
        keep = (np.isfinite(g["regret_over_floor"].to_numpy(float))
                & np.isfinite(g["mean_floor"].to_numpy(float))
                & (g["mean_floor"].to_numpy(float) > 0))
        g = g.loc[keep]
        if len(g) < 2:
            continue
        ts = g["target"].tolist()
        reg = g["mean_regret"].to_numpy(float)
        flo = g["mean_floor"].to_numpy(float)
        mor = g["regret_over_floor"].to_numpy(float)
        row = dict(zip(KEY, keys))
        row["n_targets"] = len(g)
        
        row["mean_of_ratios"] = float(mor.mean())
        mor_ci = widest_ci(mor, ts, cl)
        row.update({f"mor_{k}": v for k, v in mor_ci.items()})
        
        row["ratio_of_means"] = float(reg.mean() / flo.mean())
        row.update(ratio_of_means_ci(reg, flo, ts, cl))
        row["mean_regret"] = float(reg.mean())
        row["mean_floor"] = float(flo.mean())
        row["both_exclude_1"] = bool(
            (mor_ci["ci_hi"] < 1 or mor_ci["ci_lo"] > 1)
            and (row["rom_ci_hi"] < 1 or row["rom_ci_lo"] > 1))
        row["verdicts_disagree"] = bool(
            ((mor_ci["ci_hi"] < 1 or mor_ci["ci_lo"] > 1))
            != ((row["rom_ci_hi"] < 1 or row["rom_ci_lo"] > 1)))
        
        if keys in fz_rof.index:
            f = fz_rof.loc[keys]
            for a, b, nm in ((row["mean_of_ratios"], f["mean"], "mean"),
                             (mor_ci["ci_lo"], f["ci_lo"], "ci_lo"),
                             (mor_ci["ci_hi"], f["ci_hi"], "ci_hi")):
                if abs(float(a) - float(b)) > TOL_WRITE:
                    raise SystemExit(f"⛔ ② {keys} {nm}"
                                     f"{a} vs {b}")
            if mor_ci["boot_unit_widest"] != f["boot_unit_widest"]:
                raise SystemExit(f"⛔ ② {keys} resampleinconsistent")
        else:
            raise SystemExit(f"⛔ ② summary no {keys}")
        rows.append(row)
    sm = pd.DataFrame(rows)
    print(f"[e04d] ✅ ② {len(sm)} summary")

    out.mkdir(parents=True, exist_ok=True)
    save_table(sm, out / "rof_two_aggregations.csv", float_fmt="%.6f")
    save_table(pt, out / "per_target_level_recomputed.csv", float_fmt="%.9f")

    show = sm[KEY + ["n_targets", "mean_of_ratios", "mor_ci_lo", "mor_ci_hi",
                     "ratio_of_means", "rom_ci_lo", "rom_ci_hi",
                     "both_exclude_1", "verdicts_disagree"]]
    pd.set_option("display.width", 260)
    print(f"\n=== R-C `regret_over_floor` {scope} 20 ===")
    print(show.round(4).to_string(index=False))
    nd = int(sm["verdicts_disagree"].sum())
    print(f"\n⭐ {nd}/{len(sm)}")
    print(f"⭐ 1 {int(sm['both_exclude_1'].sum())}/{len(sm)}")
    print("⛔ mean_of_ratios ratio_of_means"
          "**** ⛔")
    if with_l3:
        print("⛔ 123 gin / gin ⛔")
        print("⛔ `note_8/59` **** ⛔ `62` §2")
    print(f"⇒ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
