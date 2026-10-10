#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits  # noqa: E402

apply_thread_limits()

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from e04_level_dissociation import (SELECTORS, EVAL_PROTOCOL, CLUSTERS,  # noqa: E402
                                    BOOT_UNITS, impute_collapse, load_target,
                                    widest_ci)
from e04_random_floor import ratio_of_means_ci, TOL_REGRET  # noqa: E402
from e17_r2_axis import FROZEN20  # noqa: E402

ALL_METRICS = ["spearman", "kendall", "enrich5", "enrich10"]
RANKING = ["spearman", "kendall"]           
OUT = ROOT / "results" / "cross_target" / "e04b_random_floor_enrich"


def collect() -> pd.DataFrame:
    rows = []
    for slug in FROZEN20:
        d = load_target(slug)
        d, _n, unexp = impute_collapse(d)
        if unexp:
            raise SystemExit(f"⛔ {slug}: NaN {unexp[:3]}")
        agg = d.groupby(["protocol", "model"])[ALL_METRICS].mean().reset_index()
        ref = pd.read_csv(ROOT / "results" / slug / "e04_hparam_pool" / "regret.csv")
        for metric in ALL_METRICS:
            piv = agg.pivot_table(index="model", columns="protocol", values=metric)
            pool_mean = float(piv[EVAL_PROTOCOL].mean())
            n_pool = int(piv[EVAL_PROTOCOL].notna().sum())
            for selector in SELECTORS:
                r = ref[(ref.metric == metric) & (ref.selector == selector)]
                if len(r) != 1:
                    raise SystemExit(f"⛔ {slug}/{metric}/{selector}: {len(r)}")
                r = r.iloc[0]
                best = float(r["temporal_best_score"])
                rows.append({
                    "target": slug, "metric": metric, "selector": selector,
                    "pool_size": int(r["pool_size"]), "n_pool_local": n_pool,
                    
                    "chosen": r["chosen_joint"], "n_ties": int(r["n_ties_joint"]),
                    "temporal_best_model": r["temporal_best"],
                    "temporal_best": best,
                    "chosen_temporal": float(r["chosen_joint_temporal"]),
                    "regret": best - float(r["chosen_joint_temporal"]),
                    
                    "pool_mean_temporal": pool_mean,
                    "floor": best - pool_mean,
                })
    return pd.DataFrame(rows)


def summarise(pt: pd.DataFrame) -> pd.DataFrame:
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")   
    out = []
    for (metric, selector), g in pt.groupby(["metric", "selector"]):
        g = g.sort_values("target")
        tg = list(g["target"]); r = g["regret"].to_numpy(); f = g["floor"].to_numpy()
        ok = f > 0
        row = {"metric": metric, "selector": selector, "n_targets": len(g),
               "mean_regret": float(r.mean()), "mean_floor": float(f.mean()),
               "min_floor": float(f.min()),
               "n_floor_nonpositive": int((~ok).sum()),
               "mean_of_ratios": float(np.mean(r[ok] / f[ok])),
               "ratio_of_means": float(r.mean() / f.mean()),
               "n_worse_than_random": int((r > f).sum()),
               "max_per_target_ratio": float(np.max(r[ok] / f[ok])),
               "max_n_ties": int(g["n_ties"].max())}
        row.update({f"mor_{k}": v for k, v in
                    widest_ci(r[ok] / f[ok], [t for t, o in zip(tg, ok) if o], cl).items()})
        row.update(ratio_of_means_ci(r, f, tg, cl))
        lo1, hi1 = row.get("mor_ci_lo"), row.get("mor_ci_hi")
        lo2, hi2 = row["rom_ci_lo"], row["rom_ci_hi"]
        row["both_exclude_1"] = bool((hi1 < 1 or lo1 > 1) and (hi2 < 1 or lo2 > 1))
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pt = collect()
    sm = summarise(pt)

    
    frozen = pd.read_csv(ROOT / "results" / "cross_target" / "e04_random_floor" / "summary.csv")
    print("=== ① metrics e04_random_floor = %.2e ===" % TOL_REGRET)
    bad = 0
    for _, fr in frozen.iterrows():
        mine = sm[(sm.metric == fr.metric) & (sm.selector == fr.selector)].iloc[0]
        for col, fcol in [("mean_regret", "mean_regret"), ("mean_floor", "mean_floor"),
                          ("mean_of_ratios", "mean_of_ratios"), ("ratio_of_means", "ratio_of_means")]:
            dv = abs(float(mine[col]) - float(fr[fcol]))
            lim = TOL_REGRET if col.startswith("mean_") and "ratio" not in col else 5e-3
            flag = "✅" if dv <= lim else "⛔"
            if dv > lim:
                bad += 1
            print(f"{flag} {fr.metric:9s}/{fr.selector:15s} {col:15s} {dv:.3e} ( {lim:.1e})")
    if bad:
        print(f"\n⛔⛔ {bad} ⇒ enrich")
        return 1
    print("⇒ ✅ \n")

    
    assert sorted(frozen.metric.unique()) == ["kendall", "spearman"]
    print("✅ ② spearman/kendall\n")

    from pparbench.io import save_table
    save_table(pt, OUT / "per_target.csv")
    save_table(sm, OUT / "summary.csv", float_fmt="%.6f")
    cols = ["metric", "selector", "mean_regret", "mean_floor", "mean_of_ratios",
            "mor_ci_lo", "mor_ci_hi", "ratio_of_means", "rom_ci_lo", "rom_ci_hi",
            "both_exclude_1", "n_worse_than_random", "max_n_ties"]
    print("=== metrics = enrichment = ===")
    pd.set_option("display.width", 250)
    print(sm[cols].round(3).to_string(index=False))
    e = sm[sm.metric.str.startswith("enrich")]
    print(f"\n⭐ enrichment 1 {int(e.both_exclude_1.sum())}/4")
    print(f"   ⇒ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
