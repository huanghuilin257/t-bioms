#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402

from pparbench.config import load_protocol                          # noqa: E402
from pparbench.io import save_table, write_run_meta                 # noqa: E402

from e20_two_stage import _boot_ci                                  # noqa: E402
from e14_loto_source import (                                       # noqa: E402
    COMPARISONS, METRICS, POOL_TREES, BOOT_SEED, CONSERVATIVE_INFLATION,
    build_loto, IN_CSV as E14_IN,
)

DECLARED_IN = "manuscript protocol declaration"
IN_DIR = ROOT / "results" / "cross_target" / "e14_loto_source"
IN_TESTS = IN_DIR / "paired_tests.csv"
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"

POOLS = ["trees4", "all7"]        
REF_FMT = "%.6g"                  


def _same(got: float, want: float) -> bool:
    """Documentation."""
    return float(REF_FMT % got) == float(want)


def _pairs(lo: pd.DataFrame, pool: str, metric: str, a: str, b: str):
    """Documentation."""
    sub = lo[(lo["pool"] == pool) & (lo["metric"] == metric)]
    wide = sub.pivot_table(index="target", columns="selector", values="regret")
    if a not in wide.columns or b not in wide.columns:
        return None, None
    pair = wide[[a, b]].dropna()
    return (pair[b] - pair[a]).to_numpy(dtype=float), list(pair.index)


def calibrate(lo: pd.DataFrame, tests: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    rng = np.random.default_rng(BOOT_SEED)
    ref = tests.set_index(["hypothesis", "pool", "metric"])
    bad = []
    for pool in POOLS:
        for metric in METRICS:
            for hid, a, b, *_ in COMPARISONS:
                delta, _ = _pairs(lo, pool, metric, a, b)
                if delta is None:
                    continue
                ci = _boot_ci(delta, None, rng)     
                key = (hid, pool, metric)
                if key not in ref.index:
                    bad.append(f"{hid}/{pool}/{metric}: no")
                    continue
                r = ref.loc[key]
                for col, got in (("delta_mean", ci["delta_mean"]),
                                 ("ci_lo", ci["ci_lo"]),
                                 ("ci_hi", ci["ci_hi"])):
                    if not _same(got, float(r[col])):
                        bad.append(f"{hid}/{pool}/{metric} {col}: "
                                   f"{got:.10g} ≠ {float(r[col]):.10g}")
    return (not bad), bad


def main() -> int:
    for p in (E14_IN, IN_TESTS):
        if not p.exists():
            print(f"[e14b] !! {p.relative_to(ROOT)} run first scripts/e14_loto_source.py")
            return 1
    
    prot = load_protocol()
    tab = pd.read_csv(E14_IN)
    lo = pd.concat([build_loto(tab, pool_name="trees4", pool=POOL_TREES),
                    build_loto(tab, pool_name="all7", pool=list(prot.models))],
                   ignore_index=True)
    tests = pd.read_csv(IN_TESTS)
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")
    print(f"[e14b] {E14_IN.name} lo {len(lo)} ,"
          f"{lo.target.nunique()} target")

    absent = sorted(set(lo.target) - set(cl.index))
    if absent:
        print(f"[e14b] !! target {CLUSTERS.name} resampletarget"
              f"{absent}")
        return 1

    
    ok, bad = calibrate(lo, tests)
    if not ok:
        print(f"[e14b] !! failed {len(bad)}"
              f"Δ_t e14 inconsistent resample")
        for m in bad[:20]:
            print(f"    {m}")
        return 1
    print(f"[e14b] {len(tests)} delta_mean/ci_lo/ci_hi"
          f"{IN_TESTS.name} {REF_FMT} consistent ⇒ target"
          f"resample")

    
    rows = []
    for pool in POOLS:
        for metric in METRICS:
            for hid, a, b, *_ in COMPARISONS:
                delta, targets = _pairs(lo, pool, metric, a, b)
                if delta is None or len(delta) < 2:
                    continue
                head = tests[(tests.hypothesis == hid) & (tests["pool"] == pool)
                             & (tests.metric == metric)]
                is_head = bool(head["is_headline"].iloc[0]) if len(head) else False
                post_hoc = (bool(head["contaminated_post_hoc"].iloc[0])
                            if len(head) else False)
                for unit_name, col in (("target", None),
                                       ("cluster_fine", "cluster_fine"),
                                       ("cluster_coarse", "cluster_coarse")):
                    units = (None if col is None
                             else np.array([cl.loc[t, col] for t in targets]))
                    ci = _boot_ci(delta, units, np.random.default_rng(BOOT_SEED))
                    rows.append({
                        "hypothesis": hid, "pool": pool, "metric": metric,
                        "arm_a": a, "arm_b": b, "boot_unit": unit_name,
                        "n_targets": int(len(delta)), "n_units": ci["n_units"],
                        "delta_mean": ci["delta_mean"],
                        "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                        "ci_lo_cons": ci["ci_lo_cons"], "ci_hi_cons": ci["ci_hi_cons"],
                        "ci_width": ci["ci_hi"] - ci["ci_lo"],
                        "excludes_zero": bool(ci["ci_lo"] > 0 or ci["ci_hi"] < 0),
                        "excludes_zero_cons": bool(ci["ci_lo_cons"] > 0
                                                   or ci["ci_hi_cons"] < 0),
                        "is_headline": is_head,
                        "contaminated_post_hoc": post_hoc,
                    })
    out = pd.DataFrame(rows)

    base = (out[out.boot_unit == "target"]
            .set_index(["hypothesis", "pool", "metric"])["ci_width"])
    out["width_vs_target"] = [
        round(r.ci_width / base.loc[(r.hypothesis, r.pool, r.metric)], 4)
        for r in out.itertuples()]

    d = ROOT / "results" / "cross_target" / "e14_cluster_bootstrap"
    d.mkdir(parents=True, exist_ok=True)
    save_table(out, d / "e14_paired_ci_by_unit.csv")

    pd.set_option("display.width", 240)
    cols = ["hypothesis", "metric", "boot_unit", "n_units", "delta_mean",
            "ci_lo", "ci_hi", "ci_lo_cons", "ci_hi_cons",
            "excludes_zero", "excludes_zero_cons", "width_vs_target"]
    print(f"\n[e14b] {len(out)} {(d / 'e14_paired_ci_by_unit.csv').relative_to(ROOT)}")
    for pool in POOLS:
        print(f"\n=== pool = {pool} ===")
        print(out[out["pool"] == pool][cols].to_string(index=False))

    
    print("\n[e14b] target CI 0")
    flips = []
    for (hid, pool, metric), g in out.groupby(["hypothesis", "pool", "metric"],
                                              sort=True):
        gi = g.set_index("boot_unit")
        for col in ("excludes_zero", "excludes_zero_cons"):
            t = bool(gi.loc["target", col])
            for u in ("cluster_fine", "cluster_coarse"):
                if u in gi.index and bool(gi.loc[u, col]) != t:
                    flips.append((hid, pool, metric, col, u, t, bool(gi.loc[u, col])))
    if not flips:
        print("resample excludes_zero / excludes_zero_cons consistent ⇒"
              "e14 resample")
    else:
        for hid, pool, metric, col, u, t, c in flips:
            print(f"    ⚠ {hid}/{pool}/{metric} {col}: target={t} → {u}={c}  "
                  f"⇒ 93 {u}")

    w = out[out.boot_unit != "target"]["width_vs_target"]
    print(f"\n[e14b] /target {w.min():.2f}–{w.max():.2f}×"
          f"{w.median():.2f}× note_8/24 §3 sqrt(20/16)≈1.12 / sqrt(20/6)≈1.83"
          f"**** 110 e19")
    print(f"[e14b] required_targets.csv T_needed target validsamples"
          f"T_needed T_needed no"
          f"T_needed target 90")
    print(f"[e14b] H4 contaminated_post_hoc"
          f"66")

    write_run_meta(d, target="cross_target", stage="e14_cluster_bootstrap",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN,
                          "boot_seed": BOOT_SEED,
                          "conservative_inflation": CONSERVATIVE_INFLATION,
                          "calibration": "replayed target-level vs paired_tests.csv",
                          "n_flips": len(flips),
                          "source_csv": str(E14_IN.relative_to(ROOT)),
                          "boot_ci_imported_from": "scripts/e20_two_stage.py::_boot_ci"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
