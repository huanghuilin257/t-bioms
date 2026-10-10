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

from e04_hparam_pool import _pick                                   # noqa: E402
from e04_level_dissociation import (EVAL_PROTOCOL, impute_collapse,  # noqa: E402
                                    load_target)
from e17_r2_axis import FROZEN20  # noqa: E402
from pparbench.io import save_table  # noqa: E402

METRICS = ["spearman", "kendall", "enrich5", "enrich10"]
SEED_FOLDS = ["seed66", "seed67", "seed68"]
OUT = ROOT / "results" / "cross_target" / "e04c_seed_spread"


def main() -> int:
    rows = []
    for slug in FROZEN20:
        d = load_target(slug)
        d, _n, unexp = impute_collapse(d)
        if unexp:
            raise SystemExit(f"⛔ {slug}: NaN {unexp[:3]}")
        rh = d[d.protocol == "random_holdout"]
        got = sorted(rh["fold"].unique())
        if got != SEED_FOLDS:
            raise SystemExit(f"⛔ {slug}: random_holdout fold 3 {got}")
        tmp_agg = (d[d.protocol == EVAL_PROTOCOL].groupby("model")[METRICS]
                   .mean())
        for metric in METRICS:
            tmp = tmp_agg[metric].dropna()
            pool = list(tmp.index)
            tbest, _ = _pick(tmp, pool)
            best = float(tmp[tbest])
            picks, regrets, ties = [], [], []
            for fold in SEED_FOLDS:
                s = rh[rh.fold == fold].set_index("model")[metric].dropna()
                ch, nt = _pick(s, pool)
                picks.append(ch); ties.append(nt)
                regrets.append(best - float(tmp[ch]))
            
            mean_s = rh.groupby("model")[metric].mean().dropna()
            ch_mean, nt_mean = _pick(mean_s, pool)
            rows.append({
                "target": slug, "metric": metric,
                "pool_size": len(pool), "temporal_best_model": tbest,
                "chosen_3seed_mean": ch_mean, "n_ties_3seed_mean": nt_mean,
                "regret_3seed_mean": best - float(tmp[ch_mean]),
                "pick_seed66": picks[0], "pick_seed67": picks[1], "pick_seed68": picks[2],
                "n_distinct_picks": len(set(picks)),
                "mean_pick_equals_any_seed_pick": ch_mean in set(picks),
                "regret_seed_min": min(regrets), "regret_seed_max": max(regrets),
                "regret_seed_range": max(regrets) - min(regrets),
                "max_n_ties_single_seed": max(ties),
            })
    pt = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    save_table(pt, OUT / "per_target.csv")

    sm = (pt.groupby("metric")
          .agg(n_targets=("target", "size"),
               n_unstable=("n_distinct_picks", lambda s: int((s > 1).sum())),
               mean_distinct_picks=("n_distinct_picks", "mean"),
               max_distinct_picks=("n_distinct_picks", "max"),
               n_mean_pick_not_any_seed=("mean_pick_equals_any_seed_pick",
                                         lambda s: int((~s).sum())),
               mean_regret_3seed=("regret_3seed_mean", "mean"),
               mean_regret_seed_range=("regret_seed_range", "mean"),
               max_regret_seed_range=("regret_seed_range", "max"))
          .reset_index())
    save_table(sm, OUT / "summary.csv", float_fmt="%.6f")
    pd.set_option("display.width", 250)
    print("=== vs 3 mean 20 ===")
    print(sm.round(4).to_string(index=False))
    print("\n⛔ 11 regret_seed_* ''")
    print(f"⇒ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
