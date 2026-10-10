#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits  # noqa: E402

apply_thread_limits()  

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from pparbench.config import load_protocol  # noqa: E402
from pparbench.io import DATA_DIR, RESULTS_DIR, save_table, write_run_meta  # noqa: E402

from e08_bedroc_check import METRICS, fold_mean_table  # noqa: E402

RANDOM, SCAFFOLD, TEMPORAL = "random_holdout", "scaffold_cv", "temporal"
PROTOCOLS = [RANDOM, SCAFFOLD, TEMPORAL]
BUCKET_THRESHOLDS = [0, 50, 100, 150]


def _frozen20() -> list[str]:
    """Documentation."""
    man = ROOT / "configs" / "manifests" / "targets_manifest.csv"
    rows = [ln.split(",")[0].strip()
            for ln in man.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    slugs = [s for s in rows[1:] if s]          
    if rows[0].split(",")[0].strip() != "slug":
        raise SystemExit(f"[e22] targets_manifest.csv slug {rows[0]!r}")
    if "pparg" not in slugs:
        slugs.append("pparg")
    if len(slugs) != 20:
        raise SystemExit(f"[e22] 20 {len(slugs)} {slugs}")
    return sorted(slugs)


def _slugs(argv_targets: list[str] | None) -> list[str]:
    if argv_targets:
        print("[e22] ⚠ target results**** 20")
        return sorted(argv_targets)
    return _frozen20()



def winner_shift(slugs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in slugs:
        r = pd.read_csv(RESULTS_DIR / s / "p03_protocol_matrix" / "ranking.csv")
        r = r[(r["aggregation"] == "fold_mean") & (r["rank"] == 1)]
        rec = {"target": s}
        for p in PROTOCOLS:
            hit = r[r["protocol"] == p]
            if len(hit) != 1:
                raise SystemExit(f"[e22] {s}/{p} fold_mean rank==1 = {len(hit)} 1")
            rec[f"{p}__model"] = hit.iloc[0]["model"]
            rec[f"{p}__family"] = hit.iloc[0]["family"]
        rows.append(rec)
    per_target = pd.DataFrame(rows)

    pairs = [(RANDOM, SCAFFOLD), (SCAFFOLD, TEMPORAL), (RANDOM, TEMPORAL)]
    summ = []
    for a, b in pairs:
        for level in ("model", "family"):
            changed = (per_target[f"{a}__{level}"] != per_target[f"{b}__{level}"]).sum()
            summ.append({"from": a, "to": b, "level": level,
                         "n_changed": int(changed), "n_targets": len(per_target),
                         "frac_changed": changed / len(per_target)})
    return per_target, pd.DataFrame(summ)



def ad_coverage(slugs: list[str]) -> pd.DataFrame:
    """Documentation."""
    parts = []
    for s in slugs:
        f = RESULTS_DIR / s / "p05_ad" / "coverage.csv"
        if not f.exists():
            print(f"[e22] warning {s} p05_ad/coverage.csv skip")
            continue
        d = pd.read_csv(f)
        d["target"] = s
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    per_t = (raw.groupby("target")
             .agg(n_cutpoints=("out_of_domain_frac", "size"),
                  ood_min=("out_of_domain_frac", "min"),
                  ood_max=("out_of_domain_frac", "max"),
                  ood_mean=("out_of_domain_frac", "mean"),
                  nnsim_mean=("mean_nn_sim", "mean"))
             .reset_index())
    return per_t


def year_hygiene(slugs: list[str]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for s in slugs:
        fl = json.loads((RESULTS_DIR / s / "p01_curate" / "run_meta.json")
                        .read_text(encoding="utf-8"))["finalize_log"]
        n, mi = fl["n_final"], fl.get("n_missing_year", 0) or 0
        pf = pd.read_csv(RESULTS_DIR / s / "p03_protocol_matrix" / "per_fold.csv")
        pf = pf[pf["model"] == pf["model"].iloc[0]]
        tot = (pf["n_train"] + pf["n"])
        temp_tot = int(tot[pf["protocol"] == TEMPORAL].max())
        cv_tot = int(tot[pf["protocol"] != TEMPORAL].max())
        if temp_tot != n - mi:
            raise SystemExit(f"[e22] {s}: temporal {temp_tot}"
                             f"n_final−n_missing_year = {n - mi} reason")
        
        d = pd.read_csv(DATA_DIR / s / "dataset.csv")
        msk = d["year"].isna()
        rows.append({"target": s, "n_final": n, "n_missing_year": mi,
                     "missing_frac": mi / n, "cv_arm_n": cv_tot,
                     "temporal_arm_n": temp_tot, "temporal_arm_loss": n - temp_tot,
                     "y_mean_missing": d.loc[msk, "y"].mean() if mi else float("nan"),
                     "y_mean_present": d.loc[~msk, "y"].mean(),
                     "frac_nobs1_missing": (d.loc[msk, "n_obs"] == 1).mean() if mi else float("nan"),
                     "frac_nobs1_present": (d.loc[~msk, "n_obs"] == 1).mean(),
                     "y_sd_all": d["y"].std(),
                     "y_sd_temporal_arm": d.loc[~msk, "y"].std(),
                     "y_sd_narrowing": d.loc[~msk, "y"].std() / d["y"].std() - 1})
    return pd.DataFrame(rows).sort_values("missing_frac", ascending=False)


def winner_shift_mcnemar(per_target: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    from scipy.stats import binomtest
    rows = []
    for level in ("model", "family"):
        x = per_target[f"{RANDOM}__{level}"] != per_target[f"{SCAFFOLD}__{level}"]
        y = per_target[f"{SCAFFOLD}__{level}"] != per_target[f"{TEMPORAL}__{level}"]
        b = int((~x & y).sum())   
        c = int((x & ~y).sum())   
        p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
        rows.append({"level": level, "n_targets": len(per_target),
                     "n_changed_rand_to_scaf": int(x.sum()),
                     "n_changed_scaf_to_temp": int(y.sum()),
                     "discordant_only_temporal": b, "discordant_only_scaffold": c,
                     "mcnemar_exact_p": p})
    return pd.DataFrame(rows)


def ad_rollup(slugs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for s in slugs:
        f = RESULTS_DIR / s / "p05_ad" / "monotonicity.csv"
        if not f.exists():
            print(f"[e22] warning {s} p05_ad/monotonicity.csv skip")
            continue
        d = pd.read_csv(f)
        d["target"] = s
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)

    summ = []
    for thr in BUCKET_THRESHOLDS:
        d = raw[raw["n_high_sim_bucket"] >= thr]
        summ.append({
            "min_n_high_sim_bucket": thr, "n_combos": len(d),
            "n_endpoints_increasing": int(d["endpoints_increasing"].sum()),
            "frac_endpoints_increasing": d["endpoints_increasing"].mean(),
            "n_monotone": int(d["monotone_increasing"].sum()),
            "frac_monotone": d["monotone_increasing"].mean(),
            "endpoint_gap_mean": d["endpoint_gap"].mean(),
            "endpoint_gap_median": d["endpoint_gap"].median(),
        })
    return raw, pd.DataFrame(summ)



def protocol_difficulty(slugs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Documentation."""
    raw = pd.concat([fold_mean_table(s).assign(target=s) for s in slugs],
                    ignore_index=True)
    rows = []
    for agg in ("max", "mean"):
        for m in METRICS:
            p = raw.pivot_table(index="target", columns="protocol", values=m, aggfunc=agg)
            base = p[RANDOM]
            for other in (SCAFFOLD, TEMPORAL):
                delta = p[other] - base
                rel = delta / base
                rows.append({
                    "pool_agg": agg, "metric": m, "vs": other, "n_targets": len(p),
                    "mean_random": base.mean(), "mean_other": p[other].mean(),
                    "delta_mean": delta.mean(), "delta_median": delta.median(),
                    "n_negative": int((delta < 0).sum()),
                    "rel_median": rel.median(),
                    "wilcoxon_p": wilcoxon(p[other], base, zero_method="zsplit").pvalue,
                })
    return raw, pd.DataFrame(rows)


def metric_sensitivity(raw: pd.DataFrame, agg: str = "max") -> pd.DataFrame:
    """Documentation."""
    rows = []
    for other in (SCAFFOLD, TEMPORAL):
        rel = {}
        for m in METRICS:
            p = raw.pivot_table(index="target", columns="protocol", values=m, aggfunc=agg)
            rel[m] = (p[other] - p[RANDOM]) / p[RANDOM]
        base = rel["spearman"]
        for m in METRICS:
            if m == "spearman":
                continue
            d = rel[m] - base
            rows.append({
                "pool_agg": agg, "vs": other, "metric": m,
                "rel_median_metric": rel[m].median(), "rel_median_spearman": base.median(),
                "diff_mean": d.mean(),
                "n_more_sensitive": int((d < 0).sum()), "n_targets": len(d),
                "wilcoxon_p": wilcoxon(rel[m], base, zero_method="zsplit").pvalue,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*")
    args = ap.parse_args()

    slugs = _slugs(args.targets)
    print(f"[e22] {len(slugs)} target {' '.join(slugs)}")
    prot = load_protocol()
    out = RESULTS_DIR / "cross_target" / "e22_scale_rollups"
    out.mkdir(parents=True, exist_ok=True)

    print("\n================ A. protocol fold_mean ================")
    wt, ws = winner_shift(slugs)
    print(ws.to_string(index=False))
    print("\n")
    for p in PROTOCOLS:
        print(f"  {p:16s} {wt[f'{p}__family'].value_counts().to_dict()}")
    print("⛔ bagging protocol CV boosting temporal bagging"
          "⑥ note_8/30 §2.3.1")
    save_table(wt, out / "winner_shift_per_target.csv")
    save_table(ws, out / "winner_shift.csv")

    print("\n---------------- A2. McNemar ----------------")
    mc = winner_shift_mcnemar(wt)
    print(mc.round(6).to_string(index=False))
    print("****target"
          "no")
    save_table(mc, out / "winner_shift_mcnemar.csv")

    print("\n================ B. applicability domain ================")
    adr, ads = ad_rollup(slugs)
    print(ads.round(4).to_string(index=False))
    print("⚠")
    save_table(ads, out / "ad_monotonicity.csv")

    print("\n---------------- B2. molecules 34–59%/48.1% ----------------")
    cov = ad_coverage(slugs)
    print(cov.round(4).to_string(index=False))
    print(f"{cov['ood_mean'].median():.4f}"
          f"targetmean {cov['ood_mean'].min():.4f}–{cov['ood_mean'].max():.4f}"
          f"{cov['ood_min'].min():.4f}–{cov['ood_max'].max():.4f}")
    save_table(cov, out / "ad_coverage.csv")

    print("\n================ D. year-missing sensitivity ================")
    yh = year_hygiene(slugs)
    cols = ["target", "n_final", "n_missing_year", "missing_frac", "temporal_arm_n",
            "y_mean_missing", "y_mean_present", "frac_nobs1_missing",
            "y_sd_all", "y_sd_temporal_arm", "y_sd_narrowing"]
    print(yh[yh.n_missing_year > 0][cols].round(4).to_string(index=False))
    print(f"20 missing year {int(yh.n_missing_year.sum())} / {int(yh.n_final.sum())}"
          f"= {yh.n_missing_year.sum()/yh.n_final.sum():.4%}；"
          f"target {int((yh.n_missing_year>0).sum())}/20")
    print("⛔ `note_8/01` §6 100% pparg"
          "20 esr1 90.1%")
    print("⛔ **** `frac_nobs1_missing` 1.00 molecules"
          "y samples")
    print("⚠ temporal y Spearman **B **"
          "note_8/20 §7.1 8 7 |narrowing| ≤ 0.81%"
          "** esr1 −8.36%**")
    save_table(yh, out / "year_hygiene.csv")

    print("\n================ C. protocol metrics ================")
    craw, pdiff = protocol_difficulty(slugs)
    print(pdiff[pdiff.pool_agg == "max"].round(6).to_string(index=False))
    print("⛔ B/A 7 model + max/mean"
          "main table trees4 29/57")
    save_table(pdiff, out / "protocol_difficulty.csv")

    print("\n================ C2. metrics spearman ================")
    sens = metric_sensitivity(craw)
    print(sens.round(6).to_string(index=False))
    print("⚠ enrich/bedroc rocauc_top"
          "metrics")
    save_table(sens, out / "metric_sensitivity.csv")

    print("\n================ C3. 18 ================")
    print(craw.groupby("protocol")[["n_tie_boundary5", "n_tie_boundary10"]]
          .agg(["mean", "max"]).round(3).to_string())
    print("⚠ temporal top10% enrichmenttemporal")

    write_run_meta(out, target="__cross_target__", stage="e22_scale_rollups",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"targets": slugs, "protocols": PROTOCOLS,
                          "metrics": METRICS,
                          "bucket_thresholds": BUCKET_THRESHOLDS,
                          "aggregation": "fold_mean only pooled",
                          "notes": "note_8/30 ⚠"
                                   "input p03/p05"
                                   "C B/A main table"})
    print(f"\n[e22] write {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
