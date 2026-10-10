from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pparbench.config import RESULTS_DIR, load_protocol, load_target
from pparbench.io import (figures_dir, save_table, target_results_dir,
                          write_run_meta)

PROTOCOL_LABEL = {
    "random_holdout": "Random 80/20",
    "scaffold_cv": "Scaffold 5-fold CV",
    "temporal": "Temporal (time-split)",
}
PROTOCOL_ORDER = ["random_holdout", "scaffold_cv", "temporal"]


def _read(path: Path, stage: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] {path} run first {stage} /skip")
        return None
    return pd.read_csv(path)


def build_table1(agg: pd.DataFrame) -> pd.DataFrame:
    """Documentation."""
    rows = []
    for model, g in agg.groupby("model"):
        row = {"model": model, "family": g["family"].iloc[0]}
        for p in PROTOCOL_ORDER:
            sub = g[g["protocol"] == p]
            if sub.empty:
                row[PROTOCOL_LABEL[p]] = "—"
                continue
            r = sub.iloc[0]
            row[PROTOCOL_LABEL[p]] = (f"{r['spearman_foldmean']:.3f}±"
                                      f"{r['spearman_foldstd']:.3f} "
                                      f"({r['spearman_pooled']:.3f})")
        
        rnd = g[g["protocol"] == "random_holdout"]["spearman_foldmean"]
        tmp = g[g["protocol"] == "temporal"]["spearman_foldmean"]
        row["inflation_random_minus_temporal"] = (
            float(rnd.iloc[0] - tmp.iloc[0]) if len(rnd) and len(tmp) else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["family", "model"]).reset_index(drop=True)


def _try_plot(fn, *args, label: str) -> None:
    """Documentation."""
    out = args[-1]
    try:
        fn(*args)
    except ImportError as e:
        print(f"[warn] {label} skip {e} matplotlib p06")
    except Exception as e:      
        print(f"[warn] {label} failed {type(e).__name__}: {e}")
    else:
        print(f"[p06] {label} → {out}")


def plot_fig1(agg: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = [p for p in PROTOCOL_ORDER if p in set(agg["protocol"])]
    models = sorted(agg["model"].unique())
    fig, ax = plt.subplots(figsize=(1.9 * len(protocols) + 3.2, 4.2))
    width = 0.8 / len(models)
    colors = {"bagging": "#2b6cb0", "boosting": "#c05621"}
    fam = agg.drop_duplicates("model").set_index("model")["family"].to_dict()

    for j, m in enumerate(models):
        xs, hs, es = [], [], []
        for i, p in enumerate(protocols):
            sub = agg[(agg.protocol == p) & (agg.model == m)]
            xs.append(i + (j - (len(models) - 1) / 2) * width)
            hs.append(float(sub["spearman_foldmean"].iloc[0]) if len(sub) else np.nan)
            es.append(float(sub["spearman_foldstd"].iloc[0]) if len(sub) else 0.0)
        ax.bar(xs, hs, width * 0.92, yerr=es, capsize=2, label=m,
               color=colors.get(fam.get(m), "#718096"),
               alpha=0.55 + 0.45 * (j % 2), edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels([PROTOCOL_LABEL[p] for p in protocols])
    ax.set_ylabel("Spearman ρ (fold mean ± SD)")
    ax.set_title("Model ranking depends on the validation protocol")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_fig2(buckets: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["<0.4", "0.4-0.6", ">=0.6"]
    present = [b for b in order if b in set(buckets["bucket"])] or\
              sorted(buckets["bucket"].unique())
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for model, g in buckets.groupby("model"):
        means = [g[g.bucket == b]["spearman"].mean() for b in present]
        ax.plot(range(len(present)), means, marker="o", label=model)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present)
    ax.set_xlabel("Nearest-neighbour Tanimoto to training set (ECFP4)")
    ax.set_ylabel("Spearman ρ (mean over time-split folds)")
    ax.set_title("Predictive power tracks the applicability domain")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def write_claims(slug: str, agg, fam, mono, cov, ci, out: Path) -> None:
    """Documentation."""
    L: list[str] = [f"# {slug}", ""]

    L.append("## ① protocol")
    if agg is not None:
        t1 = build_table1(agg)
        infl = t1["inflation_random_minus_temporal"].dropna()
        if len(infl):
            L += [f"- randomtemporal mean {infl.mean():+.3f}"
                  f"range {infl.min():+.3f} ~ {infl.max():+.3f} {len(infl)} model",
                  f"- supported {'' if infl.mean() > 0.1 else ' '}"]
        for protocol, g in agg.groupby("protocol"):
            fm = g.loc[g["spearman_foldmean"].idxmax(), "model"]
            pl = g.loc[g["spearman_pooled"].idxmax(), "model"]
            L.append(f"- {protocol}: fold-mean ={fm} pooled ={pl}"
                     f"{' ★' if fm != pl else ''}")
    L.append("")

    L.append("## ② protocolmodel")
    if agg is not None:
        for protocol, g in agg.groupby("protocol"):
            w = g.loc[g["spearman_foldmean"].idxmax()]
            L.append(f"- {protocol} {w['model']} {w['family']}"
                     f"ρ={w['spearman_foldmean']:.3f}）")
    if fam is not None:
        L.append("")
        for _, r in fam.iterrows():
            L.append(f"- {r['protocol']}: bagging {r['bagging_win_rate']:.0%}"
                     f"（{r['bagging_wins']}/{r['n_comparisons']}），"
                     f"Δρ={r['mean_delta']:+.3f} {r['n_significant']}")
        signs = set(np.sign(fam["mean_delta"]))
        L.append(f"- protocol {' ' if len(signs) > 1 else ' '}")
    L.append("")

    L.append("## ③ AD")
    if mono is not None:
        
        if "endpoints_increasing" in mono.columns:
            n_end = int(mono["endpoints_increasing"].sum())
            L.append(f"- ** > {n_end}/{len(mono)}"
                     f"（{n_end/max(1,len(mono)):.0%}）**")
            fail = mono[~mono["endpoints_increasing"]]
            if len(fail):
                L.append(f"- failed fold {sorted(fail['fold'].unique())}"
                         f"n ="
                         f"{sorted(int(v) for v in fail['n_high_sim_bucket'].unique())}")
                ok = mono[mono["endpoints_increasing"]]
                if len(ok):
                    L.append(f"- fold n ="
                             f"{int(ok['n_high_sim_bucket'].min())} "
                             f"failed")
        n_ok = int(mono["monotone_increasing"].sum())
        L.append(f"- {n_ok}/{len(mono)}"
                 f"（{n_ok/max(1,len(mono)):.0%}）")
    if cov is not None:
        L.append(f"- molecules {cov['out_of_domain_frac'].mean():.1%}"
                 f"（range {cov['out_of_domain_frac'].min():.1%}~"
                 f"{cov['out_of_domain_frac'].max():.1%}）")
        if "T" not in cov.columns and cov["fold"].astype(str).str.startswith("T").all():
            c = cov.assign(T=cov["fold"].astype(str).str.lstrip("T").astype(int)).sort_values("T")
            m = c["mean_nn_sim"].to_numpy()
            dec = bool(np.all(np.diff(m) < 0))
            L.append(f"- meanNN {' '.join(f'{v:.3f}' for v in m)}")
            L.append(f"- {'' if dec else ' ⚠ '}")
    L.append("")

    L.append("##")
    if ci is not None:
        for protocol, g in ci.groupby("protocol"):
            L.append(f"- {protocol}: CI mean {g['ci_width'].mean():.3f} /"
                     f"max {g['ci_width'].max():.3f} CI {g['ci_high'].max():.3f}")
        t = ci[ci["protocol"] == "temporal"]
        if len(t):
            best = t.loc[t["ci_high"].idxmax()]
            
            L.append(f"- **temporal CI = {best['ci_high']:.3f}"
                     f"（{best['fold']}/{best['model']}）**"
                     f"message")
    L += ["", "##",
          "- model temporalavailable",
          "- fold-mean pooled",
          "- AD samples",
          "- meanNN",
          "- single targetresultssupported CV **"]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def single_target(slug: str, prot) -> dict:
    p3 = target_results_dir(slug, "p03_protocol_matrix", create=False)
    p4 = target_results_dir(slug, "p04_bootstrap", create=False)
    p5 = target_results_dir(slug, "p05_ad", create=False)

    agg = _read(p3 / "aggregated.csv", "p03")
    fam = _read(p4 / "family_contrast.csv", "p04")
    ci = _read(p4 / "model_ci.csv", "p04")
    buckets = _read(p5 / "buckets.csv", "p05")
    mono = _read(p5 / "monotonicity.csv", "p05")
    cov = _read(p5 / "coverage.csv", "p05")

    outdir = target_results_dir(slug, "p06_report")
    figs = figures_dir(slug)

    if agg is not None:
        t1 = build_table1(agg)
        save_table(t1, outdir / "table1_protocol_matrix.csv")
        print(f"[p06] 1:\n{t1.to_string(index=False)}")
        n_empty = int((t1[[PROTOCOL_LABEL[p] for p in PROTOCOL_ORDER
                           if PROTOCOL_LABEL[p] in t1.columns]] == "—").sum().sum())
        if n_empty:
            print(f"[warn] main table {n_empty} protocolmodel")
        _try_plot(plot_fig1, agg, figs / "fig1_protocol_matrix.png", label="1")
    if fam is not None:
        save_table(fam, outdir / "table2_family_contrast.csv")
    if buckets is not None:
        save_table(buckets, outdir / "table3_ad.csv")
        _try_plot(plot_fig2, buckets, figs / "fig2_ad_buckets.png", label="2")

    write_claims(slug, agg, fam, mono, cov, ci, outdir / "claims.md")
    print(f"[p06] {outdir / 'claims.md'}")

    write_run_meta(outdir, target=slug, stage="p06_report",
                   protocol_fingerprint=prot.fingerprint)

    
    row: dict = {"target": slug}
    if agg is not None:
        for protocol, g in agg.groupby("protocol"):
            w = g.loc[g["spearman_foldmean"].idxmax()]
            row[f"{protocol}_winner"] = w["model"]
            row[f"{protocol}_winner_family"] = w["family"]
            row[f"{protocol}_best_rho"] = round(float(w["spearman_foldmean"]), 3)
    if fam is not None:
        for _, r in fam.iterrows():
            row[f"{r['protocol']}_bagging_win_rate"] = round(float(r["bagging_win_rate"]), 3)
            row[f"{r['protocol']}_mean_delta"] = round(float(r["mean_delta"]), 3)
            
            
            if "n_significant_favoring_bagging" in r.index:
                row[f"{r['protocol']}_n_sig_bagging"] = int(r["n_significant_favoring_bagging"])
            if "n_comparisons" in r.index:
                row[f"{r['protocol']}_n_comparisons"] = int(r["n_comparisons"])
    if mono is not None and len(mono):
        
        
        row["ad_monotone_frac"] = round(float(mono["monotone_increasing"].mean()), 3)
        if "endpoints_increasing" in mono.columns:
            row["ad_endpoints_frac"] = round(float(mono["endpoints_increasing"].mean()), 3)
        row["ad_n_combos"] = int(len(mono))
    if cov is not None and len(cov):
        row["mean_ood_frac"] = round(float(cov["out_of_domain_frac"].mean()), 3)

    
    
    
    
    
    
    
    if agg is not None:
        infl = {}
        for family, gf in agg.groupby("family"):
            r = gf[gf["protocol"] == "random_holdout"]["spearman_foldmean"]
            t = gf[gf["protocol"] == "temporal"]["spearman_foldmean"]
            if len(r) and len(t):
                
                infl[family] = float(r.max() - t.max())
        for family, v in infl.items():
            row[f"inflation_{family}"] = round(v, 3)
        if "bagging" in infl and "boosting" in infl:
            row["inflation_gap_boosting_minus_bagging"] = round(
                infl["boosting"] - infl["bagging"], 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="message")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target")
    g.add_argument("--targets", nargs="+")
    args = ap.parse_args()

    prot = load_protocol()
    slugs = [args.target] if args.target else args.targets
    for s in slugs:
        load_target(s)          

    rows = []
    for s in slugs:
        print(f"\n[p06] ===== {s} =====")
        rows.append(single_target(s, prot))

    if len(rows) > 1:
        cross = pd.DataFrame(rows)
        outdir = RESULTS_DIR / "cross_target"
        save_table(cross, outdir / "summary.csv")
        print(f"\n[p06] target {outdir / 'summary.csv'}")
        n = len(cross)

        if "temporal_winner_family" in cross.columns:
            vc = cross["temporal_winner_family"].value_counts()
            print(f"[p06] temporal: {vc.to_dict()}")

        
        
        
        rc, tc = "random_holdout_mean_delta", "temporal_mean_delta"
        if rc in cross.columns and tc in cross.columns:
            sub = cross.dropna(subset=[rc, tc])
            flip = sub[(sub[rc] < 0) & (sub[tc] > 0)]
            same = sub[(sub[rc] > 0) & (sub[tc] > 0)]
            print(f"[p06] random boosting temporal bagging :"
                  f"{len(flip)}/{len(sub)} target {list(flip['target'])}")
            if len(same):
                print(f"[p06] !! {len(same)}/{len(sub)} targetprotocol bagging:"
                      f"{list(same['target'])}")
                print("[p06] target**supported** supported'temporal bagging '"
                      "\n[p06] 'N/N '")

        
        sig_col = "temporal_n_sig_bagging"
        if sig_col in cross.columns and "temporal_n_comparisons" in cross.columns:
            s = cross.dropna(subset=[sig_col])
            n_any = int((s[sig_col] > 0).sum())
            detail = {r["target"]: "{}/{}".format(int(r[sig_col]),
                                                  int(r["temporal_n_comparisons"]))
                      for _, r in s.iterrows()}
            print(f"[p06] temporal bagging target: {n_any}/{len(s)}")
            print(f"[p06] target /: {detail}")

        if "ad_endpoints_frac" in cross.columns:
            print(f"[p06] AD target :"
                  f"{dict(zip(cross['target'], cross['ad_endpoints_frac']))}")

        
        gap = "inflation_gap_boosting_minus_bagging"
        if gap in cross.columns:
            s = cross.dropna(subset=[gap])
            n_pos = int((s[gap] > 0).sum())
            print(f"\n[p06] boosting − bagging = boosting :")
            for _, r in s.iterrows():
                print(f"        {r['target']:<9} bagging {r.get('inflation_bagging'):>6} "
                      f"boosting {r.get('inflation_boosting'):>6} {r[gap]:+.3f}")
            print(f"[p06] {n_pos}/{len(s)} target boosting")
            if n_pos == len(s) and len(s) > 1:
                print("[p06] ⇒ ****target ''"
                      "\n[p06] CV"
                      "\n[p06] target bagging CV")

        print(f"\n[p06] {n} target"
              "\n[p06] ②"
              "message"
              "\n[p06] ' k/N target'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
