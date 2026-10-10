#!/usr/bin/env python
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits                  # noqa: E402

apply_thread_limits()  

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

from pparbench.config import load_protocol                          # noqa: E402
from pparbench.io import (RESULTS_DIR, save_table,                   # noqa: E402
                          write_run_meta)


from e08_bedroc_check import METRICS, _metrics                      # noqa: E402
from e01_regret_analysis import POOL_TREES, SELECTORS, TEMPORAL     # noqa: E402
from e10_temporal_inner_selection import (CLEAN_METRICS, _cut_year,  # noqa: E402
                                          _load, _pick)

N_BOOT_DEFAULT = 2000
SEED = 20260825          
OPPONENTS = SELECTORS + ["random_pick"]
E10_DIR = RESULTS_DIR / "cross_target" / "e10_temporal_inner_selection"
OUT_DIR = RESULTS_DIR / "cross_target" / "e13_paired_bootstrap"




IDENTITY_ATOL = 5e-5



def _block(g: pd.DataFrame, pool: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Documentation."""
    avail = [m for m in pool if m in set(g["model"])]
    sub = {m: gm for m, gm in g.groupby("model", sort=False) if m in avail}
    ref = sub[avail[0]]
    ids = ref["mol_id"].to_numpy()          
    y = ref["y_true"].to_numpy(float)
    cols, P = [], []
    for m in avail:
        s = sub[m].set_index("mol_id")["y_pred"]
        if not s.index.is_unique or not np.isin(ids, s.index.to_numpy()).all():
            continue
        cols.append(m)
        P.append(s.reindex(ids).to_numpy(float))
    M = np.column_stack(P)
    keep = np.isfinite(M).all(axis=1) & np.isfinite(y)
    return y[keep], M[keep], cols


def cell_blocks(slug: str, pred: pd.DataFrame, year: pd.Series, D: str, E: str,
                all_pool: list[str]) -> dict:
    """Documentation."""
    t = pred[pred["protocol"] == TEMPORAL].copy()
    t["year"] = t["mol_id"].map(year)
    yD, yE = _cut_year(D), _cut_year(E)
    sel = t[(t["fold"] == D) & (t["year"] >= yD) & (t["year"] < yE)]
    ev = t[t["fold"] == E]
    
    assert sel["year"].max() < yE <= ev["year"].min(), f"{slug} {D}->{E}"
    blocks = {"eval": _block(ev, all_pool), "sel": _block(sel, all_pool)}
    for s in SELECTORS:
        g = pred[pred["protocol"] == s]
        blocks[s] = [_block(gf, all_pool) for _, gf in g.groupby("fold")]
    return blocks


def _score(blk, idx: np.ndarray | None) -> dict[str, dict[str, float]]:
    y, P, mods = blk
    if idx is not None:
        y, P = y[idx], P[idx]
    return {m: _metrics(y, P[:, j]) for j, m in enumerate(mods)}


def _fold_mean(scores_per_fold: list[dict]) -> dict[str, dict[str, float]]:
    """Documentation."""
    rows = []
    for i, sc in enumerate(scores_per_fold):
        for m, d in sc.items():
            rows.append({"model": m, "fold": i,
                         **{k: d.get(k, np.nan) for k in METRICS}})
    per_fold = pd.DataFrame(rows)
    agg = per_fold.groupby("model")[METRICS].mean()
    return {m: {k: float(v) for k, v in row.items()}
            for m, row in agg.iterrows()}



def replicate(blocks: dict, pools: dict[str, list[str]], order: list[str],
              rng: np.random.RandomState | None) -> dict:
    """Documentation."""
    def ridx(n: int) -> np.ndarray | None:
        return None if rng is None else rng.randint(0, n, n)

    ev_scores = _score(blocks["eval"], ridx(len(blocks["eval"][0])))
    sel_scores = _score(blocks["sel"], ridx(len(blocks["sel"][0])))
    selsc = {s: _fold_mean([_score(b, ridx(len(b[0]))) for b in blocks[s]])
             for s in SELECTORS}

    out = {}
    for poolname, pool in pools.items():
        for metric in METRICS:
            avail = [m for m in pool if m in ev_scores
                     and np.isfinite(ev_scores[m].get(metric, np.nan))]
            if len(avail) < 2:
                continue
            truth = {m: ev_scores[m][metric] for m in avail}
            tbest_m = max(truth, key=lambda m: truth[m])
            tbest = truth[tbest_m]
            rec = {
                "truth_best_model": tbest_m,
                
                "random_pick": float(np.mean([tbest - truth[m] for m in avail])),
            }
            ch, _nt = _pick(sel_scores, avail, metric, order)
            rec["forward"] = float(tbest - truth[ch]) if ch in truth else np.nan
            rec["forward_chosen"] = ch
            for s in SELECTORS:
                c2, _n2 = _pick(selsc[s], avail, metric, order)
                rec[s] = float(tbest - truth[c2]) if c2 in truth else np.nan
                rec[f"{s}_chosen"] = c2
            out[(poolname, metric)] = rec
    return out



def run_target(slug: str, D: str, E: str, n_boot: int, seed: int,
               all_pool: list[str], order: list[str]) -> tuple[dict, np.ndarray]:
    """Documentation."""
    pred, year = _load(slug)
    blocks = cell_blocks(slug, pred, year, D, E, all_pool)
    pools = {"trees4": POOL_TREES, "all": all_pool}
    pkeys, mkeys = list(pools), METRICS

    point = replicate(blocks, pools, order, None)

    
    
    rng = np.random.RandomState(seed)
    boot = np.full((n_boot, len(pkeys), len(mkeys), len(OPPONENTS)), np.nan)
    for b in range(n_boot):
        rep = replicate(blocks, pools, order, rng)
        for pi, pn in enumerate(pkeys):
            for mi, mt in enumerate(mkeys):
                r = rep.get((pn, mt))
                if r is None:
                    continue
                for oi, op in enumerate(OPPONENTS):
                    boot[b, pi, mi, oi] = r["forward"] - r[op]
    return point, boot



def self_check(slugs: list[str], cells: dict, all_pool: list[str],
               order: list[str]) -> pd.DataFrame:
    """Documentation."""
    ref = pd.read_csv(E10_DIR / "e10_selector_comparison.csv")
    ref = ref[ref["is_canonical"]]
    rows = []
    for slug in slugs:
        D, E = cells[slug]
        pred, year = _load(slug)
        blocks = cell_blocks(slug, pred, year, D, E, all_pool)
        point = replicate(blocks, {"trees4": POOL_TREES, "all": all_pool},
                          order, None)
        for (pn, mt), r in point.items():
            m = ref[(ref["target"] == slug) & (ref["pool"] == pn)
                    & (ref["metric"] == mt)]
            if m.empty:
                rows.append({"target": slug, "pool": pn, "metric": mt,
                             "quantity": "-", "e13": np.nan, "e10": np.nan,
                             "abs_diff": np.nan, "ok": False,
                             "note": "e10 no"})
                continue
            m = m.iloc[0]
            for q, col in [("forward", "forward_regret"),
                           ("random_pick", "random_pick_regret")] +\
                          [(s, f"{s}_regret") for s in SELECTORS]:
                a, b = r[q], float(m[col])
                d = abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
                ok = bool(np.isfinite(d) and d <= IDENTITY_ATOL) or\
                    (not np.isfinite(a) and not np.isfinite(b))
                rows.append({"target": slug, "pool": pn, "metric": mt,
                             "quantity": q, "e13": a, "e10": b,
                             "abs_diff": d, "ok": ok, "note": ""})
    return pd.DataFrame(rows)



def _pct_ci(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def aggregate(point_all: dict, boot_all: dict, slugs: list[str],
              n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Documentation."""
    pkeys = ["trees4", "all"]
    B = np.stack([boot_all[s] for s in slugs], axis=1)   # [n_boot, T, P, M, O]
    T = len(slugs)
    gr = np.random.RandomState(seed + 1)

    rows, per_rows = [], []
    for pi, pn in enumerate(pkeys):
        for mi, mt in enumerate(METRICS):
            
            nf = [point_all[s].get((pn, mt), {}).get("random_pick", np.nan)
                  for s in slugs]
            nf_mean = float(np.nanmean(nf)) if np.any(np.isfinite(nf)) else np.nan
            for oi, op in enumerate(OPPONENTS):
                d_t = np.array([point_all[s].get((pn, mt), {}).get("forward", np.nan)
                                - point_all[s].get((pn, mt), {}).get(op, np.nan)
                                for s in slugs], dtype=float)
                for s, dv in zip(slugs, d_t):
                    per_rows.append({"target": s, "pool": pn, "metric": mt,
                                     "opponent": op, "delta": dv})
                n_ok = int(np.isfinite(d_t).sum())
                n_tie = int((np.isfinite(d_t) & (d_t == 0.0)).sum())
                pt = float(np.nanmean(d_t)) if n_ok else np.nan

                
                meas = np.nanmean(B[:, :, pi, mi, oi], axis=1)
                lo_m, hi_m = _pct_ci(meas)

                
                
                
                ti = gr.randint(0, T, (n_boot, T))
                ri = gr.randint(0, n_boot, (n_boot, T))
                gen = np.nanmean(B[ri, ti, pi, mi, oi], axis=1)
                lo_g, hi_g = _pct_ci(gen)

                worst = max(abs(lo_g), abs(hi_g)) if np.isfinite(lo_g) else np.nan
                rows.append({
                    "pool": pn, "metric": mt, "opponent": op,
                    "clean_metric": mt in CLEAN_METRICS,
                    "n_targets": n_ok, "n_tie_targets": n_tie,
                    "delta_mean": pt,
                    "delta_median": float(np.nanmedian(d_t)) if n_ok else np.nan,
                    "ci_lo_measurement": lo_m, "ci_hi_measurement": hi_m,
                    "ci_lo_generalization": lo_g, "ci_hi_generalization": hi_g,
                    "noise_floor_mean": nf_mean,
                    "abs_ci_bound_generalization": worst,
                    "pct_of_noise_floor": (100.0 * worst / nf_mean)
                    if np.isfinite(worst) and nf_mean else np.nan,
                    "favors": ("forward" if np.isfinite(pt) and pt < 0
                               else "opponent" if np.isfinite(pt) and pt > 0
                               else "tie"),
                })
    return pd.DataFrame(rows), pd.DataFrame(per_rows)


# ------------------------------------------------------------------ run_meta
def _write_meta(prot, slugs, args, *, n_identity: int, post_hoc: bool) -> None:
    """Documentation."""
    extra = {
        "declared_in": "manuscript bootstrap declaration",
        "targets": slugs,
        "n_boot": args.n_boot, "seed": args.seed,
        "opponents": OPPONENTS, "metrics": METRICS,
        "ci": "percentile 95"
              "ci_*_measurement molecules ci_*_generalization targetmolecules",
        "statistic": "Δ_t = forward_regret − opponent_regret targetmean"
                     "Δ=0",
        "supersedes": "e10 sign_tests n=win+loss"
                      "16 p",
        "identity_check": f"{n_identity}/{n_identity} e10 canonical"
                          f"consistent atol {IDENTITY_ATOL} ="
                          f"⛔ consistent e10 per_target save_table"
                          f"%.4f 114",
        "limitation": "test data p03 seed"
                      "training seed e02",
        "ef_order_dependence": "enrich{5,10} input"
                               "_block CSV pivot_table",
        "tie_boundary": "_fold_mean pandas .mean() np.mean(list)"
                        "~1e-15 model",
    }
    if post_hoc:
        extra["run_meta_written_post_hoc"] = (
            "CSV metadata"
            "write_run_meta CSV meta"
            "--write-meta-only ** bootstrap**"
            "seed CSV mtime"
            "results meta results"
            "identity_check.csv")
    write_run_meta(OUT_DIR, target="__cross_target__",
                   stage="e13_paired_bootstrap",
                   protocol_fingerprint=prot.fingerprint, extra=extra)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="e13 target bootstrap")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--self-check", action="store_true",
                    help="bootstrap")
    ap.add_argument("--write-meta-only", action="store_true",
                    help="run_meta.json bootstrap"
                         "CSV meta")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()

    prot = load_protocol()
    all_pool, order = list(prot.models), list(prot.models)

    ref = pd.read_csv(E10_DIR / "e10_selector_comparison.csv")
    ref = ref[ref["is_canonical"]]
    cells = {}
    for slug, g in ref.groupby("target"):
        if g["D"].nunique() != 1 or g["E"].nunique() != 1:
            raise SystemExit(f"{slug}: canonical (D,E) e10")
        cells[slug] = (g["D"].iloc[0], g["E"].iloc[0])
    slugs = sorted(args.targets) if args.targets else sorted(cells)
    missing = [s for s in slugs if s not in cells]
    if missing:
        raise SystemExit(f"target e10 canonical no: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[e13] target {len(slugs)} trees4/{len(POOL_TREES)} + all/{len(all_pool)}")
    print(f"[e13] metrics {len(METRICS)} {OPPONENTS}")

    if args.write_meta_only:
        need = ["delta_ci.csv", "per_target_delta.csv",
                "fraction_of_noise_floor.csv", "identity_check.csv"]
        absent = [f for f in need if not (OUT_DIR / f).exists()]
        if absent:
            raise SystemExit(f"results {OUT_DIR} meta: {absent}")
        chk = pd.read_csv(OUT_DIR / "identity_check.csv")
        if not chk["ok"].all():
            raise SystemExit("identity_check.csv inconsistent meta")
        _write_meta(prot, slugs, args, n_identity=len(chk), post_hoc=True)
        print(f"[e13] run_meta.json {len(chk)} consistent -> {OUT_DIR}")
        return 0

    chk = self_check(slugs, cells, all_pool, order)
    bad = chk[~chk["ok"]]
    save_table(chk, OUT_DIR / "identity_check.csv")
    print(f"[e13] {len(chk)} inconsistent {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string(index=False))
        raise SystemExit("bug")
    print("[e13] e10 canonical consistent"
          f"atol {IDENTITY_ATOL} ⛔ consistent e10 4")
    if args.self_check:
        return 0

    from concurrent.futures import ProcessPoolExecutor
    point_all, boot_all = {}, {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_target, s, *cells[s], args.n_boot,
                          args.seed + 1000 * i, all_pool, order): s
                for i, s in enumerate(slugs)}
        for f in futs:
            s = futs[f]
            point_all[s], boot_all[s] = f.result()
            print(f"[e13] {s}")

    ci, per = aggregate(point_all, boot_all, slugs, args.n_boot, args.seed)
    save_table(ci, OUT_DIR / "delta_ci.csv")
    save_table(per, OUT_DIR / "per_target_delta.csv")

    head = ci[ci["clean_metric"] & (ci["opponent"] != "random_pick")].copy()
    save_table(head[["pool", "metric", "opponent", "delta_mean",
                     "ci_lo_generalization", "ci_hi_generalization",
                     "noise_floor_mean", "pct_of_noise_floor"]],
               OUT_DIR / "fraction_of_noise_floor.csv")

    _write_meta(prot, slugs, args, n_identity=len(chk), post_hoc=False)
    print(f"[e13] write {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
