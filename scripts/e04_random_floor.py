#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits                  # noqa: E402

apply_thread_limits()

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

from e04_hparam_pool import _pick                                  # noqa: E402
from e04_level_dissociation import (METRICS, SELECTORS, EVAL_PROTOCOL,  # noqa: E402
                                    CLUSTERS, BOOT_UNITS, impute_collapse,
                                    load_target, widest_ci)
from e20_two_stage import BOOT_SEED, N_BOOT                         # noqa: E402
from e17_r2_axis import FROZEN20                                   # noqa: E402

OUTDIR = ROOT / "results" / "cross_target" / "e04_random_floor"

#






#





#


#


TOL_REGRET = 1.09e-4


def ratio_of_means_ci(regret: np.ndarray, floor: np.ndarray,
                      targets: list[str], cl: pd.DataFrame) -> dict:
    """Documentation."""
    out, widest = {}, None
    for unit_name, col in BOOT_UNITS:
        rng = np.random.default_rng(BOOT_SEED)
        if col is None:
            groups = {t: np.array([i]) for i, t in enumerate(targets)}
        else:
            lab = [cl.loc[t, col] for t in targets]
            groups = {u: np.array([i for i, l in enumerate(lab) if l == u])
                      for u in dict.fromkeys(lab)}
        keys = list(groups)
        vals = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            sel = np.concatenate([groups[keys[i]] for i in pick])
            fm = floor[sel].mean()
            vals[b] = regret[sel].mean() / fm if fm > 0 else np.nan
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        out[f"rom_ci_lo__{unit_name}"] = float(lo)
        out[f"rom_ci_hi__{unit_name}"] = float(hi)
        if widest is None or hi - lo > widest[0]:
            widest = (hi - lo, unit_name, float(lo), float(hi))
    _, name, lo, hi = widest
    out.update({"rom_boot_unit_widest": name, "rom_ci_lo": lo, "rom_ci_hi": hi})
    return out


def main() -> int:
    per_target, checks, discrete = [], [], []
    for slug in FROZEN20:
        d = load_target(slug)
        d, n_imp, unexp = impute_collapse(d)
        if unexp:
            print(f"⛔ {slug}: {len(unexp)} NaN {unexp[:3]}")
            return 1
        agg = d.groupby(["protocol", "model"])[METRICS].mean().reset_index()
        ref = pd.read_csv(ROOT / "results" / slug / "e04_hparam_pool" / "regret.csv")
        for metric in METRICS:
            piv = agg.pivot_table(index="model", columns="protocol", values=metric)
            tmp = piv[EVAL_PROTOCOL]
            pool = list(tmp.index)
            tbest, _ = _pick(tmp, pool)
            best = float(tmp[tbest])
            floor = best - float(tmp.mean())          
            for selector in SELECTORS:
                chosen, n_ties = _pick(piv[selector], pool)
                regret = best - float(tmp[chosen])
                per_target.append({
                    "target": slug, "metric": metric, "selector": selector,
                    "pool_size": len(pool), "n_ties": n_ties,
                    "chosen": chosen, "temporal_best_model": tbest,
                    "temporal_best": best, "chosen_temporal": float(tmp[chosen]),
                    "pool_mean_temporal": float(tmp.mean()),
                    "regret": regret, "floor": floor,
                    "n_collapsed_imputed": n_imp,
                })
                
                r = ref[(ref.metric == metric) & (ref.selector == selector)]
                if len(r) == 1:
                    r0 = r.iloc[0]
                    
                    if chosen != r0["chosen_joint"]:
                        discrete.append((slug, metric, selector, "chosen",
                                         chosen, r0["chosen_joint"]))
                    if tbest != r0["temporal_best"]:
                        discrete.append((slug, metric, selector, "temporal_best",
                                         tbest, r0["temporal_best"]))
                    if len(pool) != int(r0["pool_size"]):
                        discrete.append((slug, metric, selector, "pool_size",
                                         len(pool), int(r0["pool_size"])))
                    checks.append(abs(regret - float(r0["regret_joint"])))
    pt = pd.DataFrame(per_target)

    
    print(f"[floor] ① ⛔ {len(checks)}"
          f"chosen / temporal_best / pool_size inconsistent {len(discrete)}")
    if discrete:
        for x in discrete[:10]:
            print(f"⛔ {x[0]} {x[1]}/{x[2]} {x[3]}: ={x[4]!r} ={x[5]!r}")
        print("⛔ ⇒")
        return 1

    
    if checks:
        mx = max(checks)
        print(f"[floor] ② regret_joint {mx:.2e}"
              f"{TOL_REGRET:.2e}")
        if mx > TOL_REGRET:
            print("⛔ ⇒")
            return 1
        print(f"[floor] ✅ {len(checks)}")

    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")   
    rows = []
    for (metric, selector), g in pt.groupby(["metric", "selector"]):
        g = g.set_index("target").reindex(FROZEN20)
        rg, fl = g.regret.to_numpy(dtype=float), g.floor.to_numpy(dtype=float)
        r_mean, f_mean = float(rg.mean()), float(fl.mean())
        per_target_ratio = rg / fl                      
        row = {"metric": metric, "selector": selector,
               "mean_regret": r_mean, "mean_floor": f_mean,
               
               "mean_of_ratios": float(per_target_ratio.mean()),
               
               "ratio_of_means": r_mean / f_mean if f_mean else np.nan,
               "ratio_gap_mor_minus_rom": float(per_target_ratio.mean()
                                                - (r_mean / f_mean)),
               "max_per_target_ratio": float(per_target_ratio.max()),
               "min_floor": float(fl.min()),
               "n_worse_than_random": int((g.regret > g.floor).sum()),
               "n_targets": int(g.regret.notna().sum())}
        for name, vals in (("regret", rg), ("floor", fl),
                           ("mean_of_ratios", per_target_ratio)):
            ci = widest_ci(vals, list(g.index), cl)
            row.update({f"{name}__{k}": v for k, v in ci.items()})
        row.update(ratio_of_means_ci(rg, fl, list(g.index), cl))
        rows.append(row)
    summ = pd.DataFrame(rows)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pt.to_csv(OUTDIR / "per_target.csv", index=False)
    summ.to_csv(OUTDIR / "summary.csv", index=False)
    summ["mor_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in
                      zip(summ["mean_of_ratios__ci_lo"], summ["mean_of_ratios__ci_hi"])]
    summ["rom_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in
                      zip(summ["rom_ci_lo"], summ["rom_ci_hi"])]
    cols = ["metric", "selector", "mean_regret", "mean_floor",
            "mean_of_ratios", "mor_ci", "ratio_of_means", "rom_ci",
            "n_worse_than_random", "n_targets"]
    
    print("\n" + summ[cols].to_string(index=False,
                                      float_format=lambda x: f"{x:.3f}"))
    print(f"\n[floor] → {OUTDIR}")
    print("\n = mean_of_ratios 43:56 ratio_of_means"
          "⛔")
    print("⛔ < 1"
          "⛔ CV valid")
    print("⛔ 1 ⛔ random supported")

    
    
    
    print("\n── ──")
    n_pass = 0
    for _, r in summ.iterrows():
        mor_hits = r["mean_of_ratios__ci_lo"] <= 1.0 <= r["mean_of_ratios__ci_hi"]
        rom_hits = r["rom_ci_lo"] <= 1.0 <= r["rom_ci_hi"]
        if mor_hits or rom_hits:
            who = "message" if (mor_hits and rom_hits) else (
                "message" if mor_hits else "message")
            print(f"⛔ {r.metric}/{r.selector}: 1 {who}"
                  f"⇒ **** random")
        else:
            n_pass += 1
            print(f"✅ {r.metric}/{r.selector}: 1"
                  f"（{r['mean_of_ratios']:.3f} / {r['ratio_of_means']:.3f}）")
    if n_pass == 0:
        print("\n⭐ 4 ⇒ CV"
              "****")
        print("⛔ 'failed'"
              "message")
        print("⛔ CV 0.70–0.83 < 1"
              "1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
