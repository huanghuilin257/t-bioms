#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import json
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

from e18_assay_drift import (                                       # noqa: E402
    N_BOOT, BOOT_SEED, CONSERVATIVE, verdict,
)

DECLARED_IN = "manuscript protocol declaration"
IN_DIR = ROOT / "results" / "cross_target" / "e18_assay_drift"
IN_CSV = IN_DIR / "e18_summary.csv"
IN_META = IN_DIR / "run_meta.json"
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"

ATOL = 1e-4               
GATE_K = 16               
GATE_R = 0.60
UNITS = (("target", None), ("cluster_fine", "cluster_fine"),
         ("cluster_coarse", "cluster_coarse"))


def _boot_stat(fn, n: int, units: np.ndarray | None, rng) -> np.ndarray:
    """Documentation."""
    if units is None:
        idx = rng.integers(0, n, size=(N_BOOT, n))
        return np.array([fn(i) for i in idx])
    groups = {u: np.where(units == u)[0] for u in np.unique(units)}
    keys = list(groups)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        out[b] = fn(np.concatenate([groups[keys[i]] for i in pick]))
    return out


def _ci(dist: np.ndarray, point: float) -> dict:
    """Documentation."""
    lo, hi = (float(x) for x in np.percentile(dist, [2.5, 97.5]))
    half_e18 = max(point - lo, hi - point) * CONSERVATIVE   
    half_e14 = (hi - lo) / 2 * CONSERVATIVE                 
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo,
            "ci_lo_cons_e18": point - half_e18, "ci_hi_cons_e18": point + half_e18,
            "ci_lo_cons_e14": point - half_e14, "ci_hi_cons_e14": point + half_e14}


def calibrate(summ: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    ref = json.loads(IN_META.read_text())["verdict"]
    got = verdict(summ)
    bad = []
    for k in ("n_targets", "k_B_seen_pos", "mean_B_seen", "mean_B_all", "ratio_r",
              "ci_lo", "ci_hi", "ci_lo_conservative", "ci_hi_conservative"):
        if not np.isclose(float(got[k]), float(ref[k]), rtol=0, atol=ATOL):
            bad.append(f"{k}: {got[k]:.10g} ≠ {ref[k]:.10g}"
                       f"( {abs(got[k] - ref[k]):.2g} > {ATOL:g})")
    if got["verdict"] != ref["verdict"]:
        bad.append(f"verdict: {got['verdict']} ≠ {ref['verdict']}")
    return (not bad), bad


def main() -> int:
    for p in (IN_CSV, IN_META):
        if not p.exists():
            print(f"[e18b] !! {p.relative_to(ROOT)} run first scripts/e18_assay_drift.py")
            return 1
    summ = pd.read_csv(IN_CSV).sort_values("target").reset_index(drop=True)
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")

    absent = sorted(set(summ.target) - set(cl.index))
    if absent:
        print(f"[e18b] !! target {CLUSTERS.name} resampletarget"
              f"{absent}")
        return 1

    ok, bad = calibrate(summ)
    if not ok:
        print(f"[e18b] !! failed {len(bad)}")
        for m in bad:
            print(f"    {m}")
        return 1
    ref = json.loads(IN_META.read_text())["verdict"]
    print(f"[e18b] {ATOL:g} e18_summary.csv %.4f"
          f"ratio_r / mean_B_seen / CI / run_meta.json consistent")
    print(f"[e18b] k_pos={ref['k_B_seen_pos']}"
          f"ratio_r={ref['ratio_r']:.4f}  ci_lo_cons={ref['ci_lo_conservative']:.4f}  "
          f"⇒ {ref['verdict']} (N_BOOT={N_BOOT} e14/e19 10000)")

    b_seen = summ["B_seen"].to_numpy(dtype=float)
    b_all = summ["B_all"].to_numpy(dtype=float)
    b_strict = summ["B_seen_strict"].to_numpy(dtype=float)
    n = len(summ)
    quantities = {
        "B_seen":        (lambda i: float(b_seen[i].mean()), float(b_seen.mean())),
        "B_all":         (lambda i: float(b_all[i].mean()), float(b_all.mean())),
        "B_seen_strict": (lambda i: float(b_strict[i].mean()), float(b_strict.mean())),
        
        "ratio_r":       (lambda i: float(b_seen[i].mean() / b_all[i].mean()),
                          float(b_seen.mean() / b_all.mean())),
    }

    rows = []
    for q, (fn, point) in quantities.items():
        for unit_name, col in UNITS:
            units = (None if col is None
                     else np.array([cl.loc[t, col] for t in summ.target]))
            dist = _boot_stat(fn, n, units, np.random.default_rng(BOOT_SEED))
            r = {"quantity": q, "boot_unit": unit_name, "n_targets": n,
                 "n_units": int(len(np.unique(units))) if units is not None else n}
            r.update(_ci(dist, point))
            r["excludes_zero"] = bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)
            r["excludes_zero_cons_e18"] = bool(r["ci_lo_cons_e18"] > 0
                                               or r["ci_hi_cons_e18"] < 0)
            r["excludes_zero_cons_e14"] = bool(r["ci_lo_cons_e14"] > 0
                                               or r["ci_hi_cons_e14"] < 0)
            rows.append(r)
    out = pd.DataFrame(rows)

    base = out[out.boot_unit == "target"].set_index("quantity")["ci_width"]
    out["width_vs_target"] = [round(r.ci_width / base.loc[r.quantity], 4)
                              for r in out.itertuples()]

    d = ROOT / "results" / "cross_target" / "e18_cluster_bootstrap"
    d.mkdir(parents=True, exist_ok=True)
    save_table(out, d / "e18_ratio_ci.csv", float_fmt="%.6g")

    pd.set_option("display.width", 240)
    print(f"\n[e18b] {len(out)} {(d / 'e18_ratio_ci.csv').relative_to(ROOT)}")
    print(out[["quantity", "boot_unit", "n_units", "point", "ci_lo", "ci_hi",
               "ci_lo_cons_e18", "ci_lo_cons_e14", "excludes_zero",
               "excludes_zero_cons_e18", "width_vs_target"]].to_string(index=False))

    
    k_pos = int((b_seen > 0).sum())
    r_point = float(b_seen.mean() / b_all.mean())
    print(f"\n[e18b] §3.6 k_pos={k_pos}≥{GATE_K} ✓"
          f"ratio_r={r_point:.4f}≥{GATE_R} ✓ resample")
    flipped = []
    for _, r in out[out.quantity == "B_seen"].iterrows():
        v = ("message" if (k_pos >= GATE_K and r_point >= GATE_R
                       and r["ci_lo_cons_e18"] > 0) else "message")
        if v != ref["verdict"]:
            flipped.append(r["boot_unit"])
        print(f"{r['boot_unit']:15s} ({int(r['n_units'])} )"
              f"ci_lo_cons_e18 = {r['ci_lo_cons_e18']:+.4f}  ⇒ {v}")
    if flipped:
        print(f"⚠ {flipped} ⇒ 93"
              f"v3 assay")
    else:
        print(f"resample §3.6 {ref['verdict']} ⇒"
              f"e18 assay resample")

    w = out[out.boot_unit != "target"]["width_vs_target"]
    print(f"\n[e18b] /target {w.min():.2f}–{w.max():.2f}×"
          f"{w.median():.2f}× note_8/24 §3 sqrt(20/16)≈1.12 /"
          f"sqrt(20/6)≈1.83 ****")
    print(f"[e18b] assay 69"
          f"strict 74 B_seen_strict"
          f"`pparg` superset 67"
          f"`kcnh2` B_seen<0 target 70")

    write_run_meta(d, target="cross_target", stage="e18_cluster_bootstrap",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN, "n_boot": N_BOOT,
                          "boot_seed": BOOT_SEED, "conservative_factor": CONSERVATIVE,
                          "calibration_atol": ATOL,
                          "calibration": "verdict() vs e18 run_meta.json",
                          "gate_k": GATE_K, "gate_r": GATE_R,
                          "verdict_flipped_units": flipped,
                          "source_csv": str(IN_CSV.relative_to(ROOT))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
