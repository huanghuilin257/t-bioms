#!/usr/bin/env python3
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
from scipy.stats import binomtest, wilcoxon                        # noqa: E402

from pparbench.config import load_protocol                         # noqa: E402
from pparbench.io import (RESULTS_DIR, canonical_dataset_path,      # noqa: E402
                          save_table, target_results_dir, write_run_meta)


from e08_bedroc_check import METRICS, _metrics, fold_mean_table     # noqa: E402
from e01_regret_analysis import POOL_TREES, SELECTORS, TEMPORAL     # noqa: E402

MIN_N_DEFAULT = 100      
N_BOOT_DEFAULT = 200     


BOOT_METRICS = ["spearman", "bedroc20_top5", "rocauc_top10"]

CLEAN_METRICS = ["spearman", "kendall", "bedroc20_top5", "rocauc_top5",
                 "bedroc20_top10", "rocauc_top10"]


def _cut_year(fold: str) -> int:
    """'T2015' -> 2015。"""
    return int(str(fold).lstrip("T"))


def _load(slug: str) -> tuple[pd.DataFrame, pd.Series]:
    """Documentation."""
    p = target_results_dir(slug, "p03_protocol_matrix", create=False) / "predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    pred = pd.read_csv(p)
    year = pd.read_csv(canonical_dataset_path(slug)).set_index("mol_id")["year"]
    miss = set(pred.loc[pred["protocol"] == TEMPORAL, "mol_id"]) - set(year.dropna().index)
    assert not miss, f"{slug}: temporal {len(miss)} moleculesnoyear split logic"
    return pred, year


def _scores(g: pd.DataFrame, pool: list[str]) -> dict[str, dict[str, float]]:
    """Documentation."""
    return {m: _metrics(gm["y_true"].to_numpy(float), gm["y_pred"].to_numpy(float))
            for m, gm in g.groupby("model") if m in pool}


def _pick(scores: dict[str, dict[str, float]], avail: list[str],
          metric: str, order: list[str]) -> tuple[str | None, int]:
    """Documentation."""
    ok = [m for m in avail if m in scores and np.isfinite(scores[m].get(metric, np.nan))]
    if len(ok) < 2:
        return None, 0
    vals = {m: scores[m][metric] for m in ok}
    best = max(vals.values())
    tied = sorted([m for m in ok if vals[m] == best],
                  key=lambda m: order.index(m) if m in order else 99)
    return tied[0], len(tied)


# ------------------------------------------------------------------ A
def per_cutpoint(slug: str, pred: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    t = pred[pred["protocol"] == TEMPORAL]
    rows = []
    for (fold, model), g in t.groupby(["fold", "model"]):
        rows.append({"target": slug, "cutpoint": fold, "year": _cut_year(fold),
                     "model": model, "n_test": len(g),
                     **_metrics(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))})
    return pd.DataFrame(rows)


def champion_stability(pc: pd.DataFrame, pool: list[str], poolname: str,
                       min_n: int) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for tg, gt in pc.groupby("target"):
        g = gt[gt["model"].isin(pool) & (gt["n_test"] >= min_n)]
        for metric in METRICS:
            w = g.pivot_table(index="cutpoint", columns="model", values=metric)
            w = w.reindex(columns=[m for m in pool if m in w.columns]).dropna(how="any")
            if len(w) < 2 or w.shape[1] < 2:
                continue
            champs = w.idxmax(axis=1)
            fm_champ = w.mean(axis=0).idxmax()
            vc = champs.value_counts()
            rows.append({"target": tg, "pool": poolname, "metric": metric,
                         "n_cut": int(len(w)),
                         "n_distinct": int(champs.nunique()),
                         "modal": vc.index[0],
                         "modal_frac": float(vc.iloc[0] / len(w)),
                         "fm_champ": fm_champ,
                         "fm_hit": float((champs == fm_champ).mean()),
                         "path": "->".join(champs.sort_index().tolist())})
    return pd.DataFrame(rows)


def path_autocorr(stab: pd.DataFrame, pool: str = "trees4") -> pd.DataFrame:
    """Documentation."""
    rows = []
    for metric in BOOT_METRICS:
        a = stab[(stab["pool"] == pool) & (stab["metric"] == metric)]
        adj, allp = [], []
        for r in a.itertuples():
            p = r.path.split("->")
            if len(p) < 3:
                continue
            adj.append(np.mean([p[i] == p[i + 1] for i in range(len(p) - 1)]))
            allp.append(np.mean([p[i] == p[j] for i in range(len(p))
                                 for j in range(i + 1, len(p))]))
        if not adj:
            continue
        adj_a, all_a = np.asarray(adj), np.asarray(allp)
        d = adj_a - all_a
        k, n = int((d > 0).sum()), int((d != 0).sum())
        try:
            pw = float(wilcoxon(adj_a, all_a).pvalue)
        except ValueError:
            pw = float("nan")
        rows.append({"metric": metric, "n_targets": len(adj_a),
                     "adjacent_same_champ": float(adj_a.mean()),
                     "allpair_same_champ": float(all_a.mean()),
                     "diff": float(d.mean()), "k_adj_higher": f"{k}/{n}",
                     "p_sign": binomtest(k, n, 0.5).pvalue if n else np.nan,
                     "p_wilcoxon": pw})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B
def forward_pairs(slug: str, pred: pd.DataFrame, year: pd.Series, min_n: int,
                  all_pool: list[str]) -> pd.DataFrame:
    """Documentation."""
    t = pred[pred["protocol"] == TEMPORAL].copy()
    t["year"] = t["mol_id"].map(year)
    cuts = sorted(t["fold"].unique(), key=_cut_year)
    rows = []
    for i, D in enumerate(cuts):
        yD, gD = _cut_year(D), t[t["fold"] == D]
        for E in cuts[i + 1:]:
            yE = _cut_year(E)
            sel = gD[(gD["year"] >= yD) & (gD["year"] < yE)]   
            ev = t[t["fold"] == E]
            if sel["mol_id"].nunique() < min_n or ev["mol_id"].nunique() < min_n:
                continue
            assert sel["year"].max() < yE <= ev["year"].min(), f"{slug} {D}->{E}"
            rows.append({"target": slug, "D": D, "E": E, "gap": yE - yD,
                         "n_sel": int(sel["mol_id"].nunique()),
                         "n_eval": int(ev["mol_id"].nunique()),
                         "sel_scores": _scores(sel, all_pool),
                         "ev_scores": _scores(ev, all_pool)})
    return pd.DataFrame(rows)


def compare(fp: pd.DataFrame, fm: pd.DataFrame, pool: list[str], poolname: str,
            order: list[str]) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for r in fp.itertuples(index=False):
        fmt = fm[fm["target"] == r.target] if "target" in fm.columns else fm
        for metric in METRICS:
            avail = [m for m in pool
                     if m in r.ev_scores and np.isfinite(r.ev_scores[m].get(metric, np.nan))]
            if len(avail) < 2:
                continue
            truth = {m: r.ev_scores[m][metric] for m in avail}
            tbest_m = max(truth, key=lambda m: truth[m])
            tbest = truth[tbest_m]
            rec = {"target": r.target, "pool": poolname, "metric": metric,
                   "D": r.D, "E": r.E, "gap": r.gap,
                   "n_sel": r.n_sel, "n_eval": r.n_eval, "truth_best_model": tbest_m,
                   
                   "random_pick_regret": float(np.mean([tbest - truth[m] for m in avail]))}
            ch, nt = _pick(r.sel_scores, avail, metric, order)
            rec["forward_chosen"], rec["forward_ties"] = ch, nt
            rec["forward_regret"] = float(tbest - truth[ch]) if ch in truth else np.nan
            for s in SELECTORS:      
                sub = fmt[fmt["protocol"] == s]
                sc = {m: {metric: float(v)} for m, v in zip(sub["model"], sub[metric])
                      if np.isfinite(v)}
                c2, nt2 = _pick(sc, avail, metric, order)
                rec[f"{s}_chosen"], rec[f"{s}_ties"] = c2, nt2
                rec[f"{s}_regret"] = float(tbest - truth[c2]) if c2 in truth else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def mark_canonical(cmp: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    cmp = cmp.copy()
    cmp["is_canonical"] = False
    for _k, g in cmp.groupby(["target", "pool", "metric"]):
        sub = g[g["E"] == g["E"].max()]
        cmp.loc[sub.index[sub["D"] == sub["D"].max()], "is_canonical"] = True
    return cmp


def sign_tests(cmp: pd.DataFrame, canonical_only: bool = True) -> pd.DataFrame:
    """Documentation."""
    d = cmp[cmp["is_canonical"]] if canonical_only else cmp
    rows = []
    for (poolname, metric), g in d.groupby(["pool", "metric"]):
        r = {"pool": poolname, "metric": metric, "n_targets": int(g["target"].nunique())}
        for s in SELECTORS + ["random_pick"]:
            col = "random_pick_regret" if s == "random_pick" else f"{s}_regret"
            v = g[["forward_regret", col]].dropna()
            win = int((v["forward_regret"] < v[col]).sum())
            loss = int((v["forward_regret"] > v[col]).sum())
            n = win + loss
            r[f"fwd_beats_{s}"] = f"{win}/{n}"
            r[f"p_vs_{s}"] = binomtest(win, n, 0.5).pvalue if n else np.nan
        r["median_forward"] = float(g["forward_regret"].median())
        for s in SELECTORS:
            r[f"median_{s}"] = float(g[f"{s}_regret"].median())
        r["median_random_pick"] = float(g["random_pick_regret"].median())
        rows.append(r)
    out = pd.DataFrame(rows)
    out["_mo"] = out["metric"].map({m: i for i, m in enumerate(METRICS)})
    return out.sort_values(["pool", "_mo"]).drop(columns="_mo").reset_index(drop=True)


# ------------------------------------------------------------------ C
def bootstrap_within(slug: str, pred: pd.DataFrame, min_n: int, n_boot: int,
                     rng: np.random.RandomState) -> pd.DataFrame:
    """Documentation."""
    t = pred[(pred["protocol"] == TEMPORAL) & pred["model"].isin(POOL_TREES)]
    rows = []
    for fold, g in t.groupby("fold"):
        w = g.pivot_table(index="mol_id", columns="model", values="y_pred")
        w = w.reindex(columns=[m for m in POOL_TREES if m in w.columns])
        y = g.groupby("mol_id")["y_true"].first().reindex(w.index).to_numpy(float)
        if len(w) < min_n or w.shape[1] < 2:
            continue
        P, mods, n = w.to_numpy(float), list(w.columns), len(y)
        s0 = {m: _metrics(y, P[:, j]) for j, m in enumerate(mods)}
        pt = {}
        for metric in BOOT_METRICS:
            v = {m: s0[m][metric] for m in mods if np.isfinite(s0[m][metric])}
            pt[metric] = max(v, key=lambda m: v[m]) if len(v) >= 2 else None
        hit = {m: 0 for m in BOOT_METRICS}
        tot = {m: 0 for m in BOOT_METRICS}
        seen: dict[str, set] = {m: set() for m in BOOT_METRICS}
        for _b in range(n_boot):
            idx = rng.randint(0, n, n)          
            yb = y[idx]
            sc = {m: _metrics(yb, P[idx, j]) for j, m in enumerate(mods)}
            for metric in BOOT_METRICS:
                v = {m: sc[m][metric] for m in mods if np.isfinite(sc[m][metric])}
                if len(v) < 2:
                    continue
                c = max(v, key=lambda m: v[m])
                tot[metric] += 1
                seen[metric].add(c)
                hit[metric] += int(c == pt[metric])
        for metric in BOOT_METRICS:
            if tot[metric]:
                rows.append({"target": slug, "cutpoint": fold, "metric": metric,
                             "n_test": n, "pt_champ": pt[metric],
                             "boot_repro": hit[metric] / tot[metric],
                             "boot_distinct": len(seen[metric])})
    return pd.DataFrame(rows)


def drift_vs_noise(bs: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for metric in BOOT_METRICS:
        b = bs[bs["metric"] == metric]
        a = stab[(stab["pool"] == "trees4") & (stab["metric"] == metric)]
        m = (b.groupby("target")["boot_repro"].mean().rename("within_boot").to_frame()
             .join(a.set_index("target")["modal_frac"].rename("across_cut")).dropna())
        if m.empty:
            continue
        k, n = int((m["across_cut"] < m["within_boot"]).sum()), len(m)
        try:
            pw = float(wilcoxon(m["across_cut"], m["within_boot"]).pvalue)
        except ValueError:
            pw = float("nan")
        gap, noise = 1 - m["across_cut"], 1 - m["within_boot"]
        share = (noise / gap).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"metric": metric, "n_targets": n,
                     "mean_within_boot_repro": float(m["within_boot"].mean()),
                     "mean_across_cut_modal": float(m["across_cut"].mean()),
                     "k_across_worse_than_noise": k,
                     "p_sign": binomtest(k, n, 0.5).pvalue, "p_wilcoxon": pw,
                     "median_diff": float((m["across_cut"] - m["within_boot"]).median()),
                     "total_gap": float(gap.mean()),
                     "noise_explains": float(noise.mean()),
                     "noise_share_median": float(share.median()),
                     "residual_upper_bound": float(max(0.0, gap.mean() - noise.mean()))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="e10 /")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="A+B")
    args = ap.parse_args()

    prot = load_protocol()
    all_pool = list(prot.models)
    order = list(prot.models)          
    rng = np.random.RandomState(getattr(prot, "seed", 66))

    if args.targets:
        slugs = args.targets
    else:
        slugs = sorted(d.name for d in RESULTS_DIR.iterdir()
                       if d.is_dir() and d.name != "cross_target"
                       and (d / "p03_protocol_matrix" / "predictions.csv").exists())
    print(f"[e10] target {len(slugs)}: {slugs}")
    print(f"[e10] {all_pool}\n[e10] {POOL_TREES}")
    print(f"[e10] min_n={args.min_n}  n_boot={args.n_boot}"
          f"{' skip C ' if args.skip_bootstrap else ''}")

    pcs, fps, fms, bss, dropped = [], [], [], [], []
    for slug in slugs:
        try:
            pred, year = _load(slug)
        except FileNotFoundError:
            dropped.append((slug, "p03 predictions.csv"))
            continue
        pcs.append(per_cutpoint(slug, pred))
        fmt = fold_mean_table(slug)
        if "target" not in fmt.columns:
            fmt = fmt.assign(target=slug)
        fms.append(fmt)
        fp = forward_pairs(slug, pred, year, args.min_n, all_pool)
        n_cut = pred[pred["protocol"] == TEMPORAL]["fold"].nunique()
        if fp.empty:
            dropped.append((slug, f"no n_sel/n_eval >= {args.min_n} (D,E)"))
        else:
            fps.append(fp)
        if not args.skip_bootstrap:
            bss.append(bootstrap_within(slug, pred, args.min_n, args.n_boot, rng))
        print(f"[e10] {slug:9s} {n_cut} {len(fp)}")

    pc = pd.concat(pcs, ignore_index=True)
    fm_all = pd.concat(fms, ignore_index=True)
    fp_all = pd.concat(fps, ignore_index=True)

    out = RESULTS_DIR / "cross_target" / "e10_temporal_inner_selection"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 80)

    # ---------------- A ----------------
    stab = pd.concat([champion_stability(pc, POOL_TREES, "trees4", args.min_n),
                      champion_stability(pc, all_pool, "all", args.min_n)],
                     ignore_index=True)
    save_table(pc, out / "e10_per_cutpoint_scores.csv")
    save_table(stab, out / "e10_champion_stability.csv")

    print("\n======== A test data => ========")
    for poolname in ("trees4", "all"):
        s = stab[stab["pool"] == poolname]
        if s.empty:
            continue
        print(f"\n-- pool={poolname} --")
        print(s.groupby("metric").agg(
            n_targets=("target", "nunique"), mean_n_cut=("n_cut", "mean"),
            mean_distinct=("n_distinct", "mean"), max_distinct=("n_distinct", "max"),
            frac_unstable=("n_distinct", lambda v: (v > 1).mean()),
            mean_modal_frac=("modal_frac", "mean"), mean_fm_hit=("fm_hit", "mean"),
        ).reindex(METRICS).round(3).to_string())

    ac = path_autocorr(stab)
    save_table(ac, out / "e10_path_autocorrelation.csv")
    print("\n-- no '' --")
    print(ac.round(4).to_string(index=False))

    # ---------------- B ----------------
    cmp = pd.concat([compare(fp_all, fm_all, pool, poolname, order)
                     for poolname, pool in (("trees4", POOL_TREES), ("all", all_pool))],
                    ignore_index=True)
    cmp = mark_canonical(cmp)
    st = sign_tests(cmp, canonical_only=True)
    st_all = sign_tests(cmp, canonical_only=False)
    save_table(cmp, out / "e10_selector_comparison.csv")
    save_table(st, out / "e10_sign_tests.csv")
    save_table(st_all, out / "e10_sign_tests_all_pairs.csv")

    print("\n======== B vs (D,E) target ========")
    print(st.to_string(index=False))
    print("\n---- (D,E) ----")
    print(st_all[["pool", "metric", "n_targets", "fwd_beats_scaffold_cv",
                  "p_vs_scaffold_cv", "fwd_beats_random_holdout",
                  "fwd_beats_random_pick", "p_vs_random_pick"]].to_string(index=False))

    print("\n======== , trees4, metrics ========")
    c = cmp[cmp["is_canonical"] & (cmp["pool"] == "trees4")
            & cmp["metric"].isin(CLEAN_METRICS)]
    picks = pd.DataFrame({lab: c[col].value_counts()
                          for col, lab in (("truth_best_model", "message"),
                                           ("forward_chosen", "forward"),
                                           ("scaffold_cv_chosen", "scaffold_cv"),
                                           ("random_holdout_chosen", "random_holdout"))})
    print(picks.reindex(POOL_TREES).fillna(0).astype(int).to_string())
    print("=> forward"
          "**** ****")

    # ---------------- C ----------------
    if not args.skip_bootstrap:
        bs = pd.concat(bss, ignore_index=True)
        dvn = drift_vs_noise(bs, stab)
        save_table(bs, out / "e10_bootstrap_within_cutpoint.csv")
        save_table(dvn, out / "e10_drift_vs_noise.csv")
        print(f"\n======== C bootstrap trees4, n_boot={args.n_boot} ========")
        print(bs.groupby("metric").agg(
            n_cells=("target", "size"), n_targets=("target", "nunique"),
            mean_repro=("boot_repro", "mean"), median_repro=("boot_repro", "median"),
            mean_distinct=("boot_distinct", "mean"),
            frac_champ_changed=("boot_distinct", lambda v: (v > 1).mean()),
        ).reindex(BOOT_METRICS).round(3).to_string())
        print("\n-- --")
        print(dvn.round(4).to_string(index=False))
        print("across ≈ within => A fold-mean")
        print("across => residual_upper_bound ****")
        print("training data")

    print("\n--- ---")
    print("* **** e01/e08/e09 fold-mean")
    print("* scaffold_cv 11")
    print("fwd_beats_*")
    print("* enrich{5,10} note_8/10 §2 bedroc20/rocauc")
    print("* test data => A C")
    print("* '' ****'' e10_path_autocorrelation")
    if dropped:
        print("\n[e10] target")
        for s, why in dropped:
            print(f"    {s}: {why}")

    write_run_meta(out, target="__cross_target__", stage="e10_temporal_inner_selection",
                   protocol_fingerprint=prot.fingerprint,
                   extra={"targets": slugs, "dropped": dict(dropped),
                          "full_pool": all_pool, "pool_trees": POOL_TREES,
                          "metrics": METRICS, "boot_metrics": BOOT_METRICS,
                          "min_n": args.min_n, "n_boot": args.n_boot,
                          "skipped_bootstrap": bool(args.skip_bootstrap),
                          "regret_scope": "fold-mean",
                          "leakage_control": "year [D, E)"
                                             "year>=E",
                          "nested_cutpoints": "test data A"
                                              "C bootstrap",
                          "residual_is_upper_bound": "C training data"
                                                     "message"
                                                     "''''",
                          "ef_order_dependence": "enrich{5,10} input"
                                                 "note_8/10 §2 bedroc20/rocauc",
                          "notes": "''"
                                   "input p03 predictions.csv"})
    print(f"\n[e10] write -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
