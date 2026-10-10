#!/usr/bin/env python
"""Documentation."""
import sys, pathlib, numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e20_two_stage import BOOT_SEED, N_BOOT
from e04_level_dissociation import CLUSTERS, BOOT_UNITS
from pparbench.io import save_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "results/cross_target/e01_regret/e01_regret.csv"
OUT  = ROOT / "results/cross_target/e01b_ratio_aggregation_ci"
POOLS   = ["trees4", "all"]          




METRICS = ["spearman", "kendall", "enrich5", "enrich10"]
RANKING = {"spearman", "kendall"}    
SELS    = ["random_holdout", "scaffold_cv"]


def both_aggregations_ci(r, f, targets, cl):
    """Documentation."""
    per_unit = {}
    for name, col in BOOT_UNITS:
        rng = np.random.default_rng(BOOT_SEED)
        if col is None:
            groups = {t: np.array([i]) for i, t in enumerate(targets)}
        else:
            lab = [cl.loc[t, col] for t in targets]
            groups = {u: np.array([i for i, l in enumerate(lab) if l == u])
                      for u in dict.fromkeys(lab)}
        keys = list(groups)
        mor = np.empty(N_BOOT); rom = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            sel  = np.concatenate([groups[keys[i]] for i in pick])
            rr, ff = r[sel], f[sel]
            ok = ff > 0
            mor[b] = np.mean(rr[ok] / ff[ok]) if ok.any() else np.nan
            rom[b] = rr.mean() / ff.mean() if ff.mean() > 0 else np.nan
        per_unit[name] = (np.nanpercentile(mor, [2.5, 97.5]),
                          np.nanpercentile(rom, [2.5, 97.5]))
    widest = lambda k: max(per_unit.values(), key=lambda v: v[k][1] - v[k][0])[k]
    return widest(0), widest(1), per_unit


def main():
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")
    d  = pd.read_csv(SRC)
    rows = []
    for pool in POOLS:
        for met in METRICS:
            for sel in SELS:
                s = d[(d["pool"] == pool) & (d.metric == met)
                      & (d.selector == sel)].sort_values("target")
                assert len(s) == 20, (pool, met, sel, len(s))
                r, f, tg = s.regret.values, s.random_pick_regret.values, list(s.target)
                mor = float(np.mean(r[f > 0] / f[f > 0]))
                rom = float(r.mean() / f.mean())
                (ml, mh), (rl, rh), per_unit = both_aggregations_ci(r, f, tg, cl)
                rows.append(dict(
                    pool=pool, metric=met, selector=sel, n_target=len(s),
                    regret_mean=r.mean(), floor_mean=f.mean(),
                    mean_of_ratios=mor, mor_ci_lo=ml, mor_ci_hi=mh,
                    mor_excludes_1=bool(mh < 1),
                    ratio_of_means=rom, rom_ci_lo=rl, rom_ci_hi=rh,
                    rom_excludes_1=bool(rh < 1),
                    widest_unit_mor=max(per_unit, key=lambda k: per_unit[k][0][1]-per_unit[k][0][0]),
                ))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    save_table(out, OUT / "ratio_aggregation_ci.csv")

    
    chk = out[(out['pool'] == "trees4") & (out.metric == "spearman")].set_index("selector")
    for sel, expect in [("random_holdout", 0.937), ("scaffold_cv", 0.745)]:
        got_rom = chk.loc[sel, "ratio_of_means"]
        got_mor = chk.loc[sel, "mean_of_ratios"]
        assert abs(got_rom - expect) < 5e-4, (sel, got_rom, expect)
        print(f"{sel}: {expect} = ratio_of_means {got_rom:.4f} ✅"
              f"| 43:56 mean_of_ratios = {got_mor:.4f}")

    print(f"\n {OUT/'ratio_aggregation_ci.csv'}")
    show = ["pool","metric","selector","regret_mean","floor_mean",
            "mean_of_ratios","mor_ci_lo","mor_ci_hi","mor_excludes_1",
            "ratio_of_means","rom_ci_lo","rom_ci_hi","rom_excludes_1"]
    print(out[show].to_string(index=False))
    n = len(out)
    print(f"\n 1 {int(out.mor_excludes_1.sum())}/{n}"
          f"{int(out.rom_excludes_1.sum())}/{n}")

    rank = out[out.metric.isin(RANKING)]
    enr  = out[~out.metric.isin(RANKING)]
    print("\n metrics ⛔")
    for nm, sub in [("spearman/kendall", rank), ("enrichment enrich5/enrich10", enr)]:
        a = sub[sub['pool']=="all"]; t = sub[sub['pool']=="trees4"]
        print(f"{nm}: trees4 1 {int(t.mor_excludes_1.sum())}/{len(t)}"
              f"all7 1 {int(a.mor_excludes_1.sum())}/{len(a)}")
    print("\n⛔ trees4 1 ⇒"
          "⛔ 6–26%")
    print("✅ CV **metrics**"
          "all7 × spearman/kendall 1")
    print("⛔ enrichmentmetrics all7 × enrich ** 1**"
          "0.739–0.980 ⇒ **metrics**")
    hi = out[out.mean_of_ratios > 1]
    for _, r_ in hi.iterrows():
        print(f"⚠ > 1 {r_['pool']} × {r_.metric} × {r_.selector}"
              f"= {r_.mean_of_ratios:.3f} [{r_.mor_ci_lo:.3f},{r_.mor_ci_hi:.3f}]"
              f"⇒ 1 **** CV ⛔")


if __name__ == "__main__":
    main()
