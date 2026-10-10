#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits            # noqa: E402

apply_thread_limits(1)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from scipy import stats                                       # noqa: E402

from pparbench.config import RESULTS_DIR                       # noqa: E402

E01_DIR = RESULTS_DIR / "cross_target" / "e01_regret"
OUT_DIR = RESULTS_DIR / "e12_statistic_comparison"

POOL_TREES = ["lightgbm", "hgb", "rf", "extratrees"]
TEMPORAL = "temporal"


R_HIGH = 0.80
RATIO_BAD = 1.00


def win_prob(chosen: str, tmp: dict[str, float], pool: list[str]) -> float:
    """Documentation."""
    others = [m for m in pool if m in tmp and m != chosen]
    if not others or chosen not in tmp:
        return np.nan
    c = tmp[chosen]
    wins = sum(1.0 if c > tmp[m] else (0.5 if np.isclose(c, tmp[m]) else 0.0)
               for m in others)
    return wins / len(others)


def build_cells() -> pd.DataFrame:
    agg = pd.read_csv(E01_DIR / "e01_foldmean_all_metrics.csv")
    reg = pd.read_csv(E01_DIR / "e01_regret.csv")
    metrics = sorted(reg["metric"].unique())
    full_pool = sorted(agg["model"].unique())
    pools = {"trees4": POOL_TREES, "all": full_pool}

    rows: list[dict] = []
    for _, r in reg.iterrows():
        slug, pool_name, selector, metric = (
            r["target"], r["pool"], r["selector"], r["metric"])
        pool = pools[pool_name]
        piv = (agg[agg["target"] == slug]
               .pivot_table(index="model", columns="protocol", values=metric))
        if selector not in piv.columns or TEMPORAL not in piv.columns:
            continue
        s_all = piv[selector].dropna().to_dict()
        t_all = piv[TEMPORAL].dropna().to_dict()
        avail = [m for m in pool if m in s_all and m in t_all]
        if len(avail) < 3:
            continue
        s = np.array([s_all[m] for m in avail], dtype=float)
        t = np.array([t_all[m] for m in avail], dtype=float)

        
        if np.std(s) == 0 or np.std(t) == 0:
            pr = rho = np.nan
        else:
            pr = float(stats.pearsonr(s, t).statistic)
            rho = float(stats.spearmanr(s, t).statistic)

        ratio = (float(r["regret"]) / float(r["random_pick_regret"])
                 if float(r["random_pick_regret"]) > 0 else np.nan)

        rows.append({
            "target": slug, "pool": pool_name, "selector": selector,
            "metric": metric, "pool_size": len(avail),
            "pearson_r": round(pr, 4) if np.isfinite(pr) else np.nan,
            "spearman_rho": round(rho, 4) if np.isfinite(rho) else np.nan,
            "win_prob": round(win_prob(r["chosen"], t_all, pool), 4),
            "regret": round(float(r["regret"]), 4),
            "random_pick_regret": round(float(r["random_pick_regret"]), 4),
            "ratio": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            "chosen": r["chosen"], "temporal_best_model": r["temporal_best_model"],
            
            "top_gap": round(float(np.sort(t)[-1] - np.sort(t)[-2]), 4),
            "spread": round(float(t.max() - t.min()), 4),
        })
    return pd.DataFrame(rows)


def write_verdict(cells: pd.DataFrame) -> None:
    ok = cells.dropna(subset=["pearson_r", "ratio"])
    dec = ok[(ok.pearson_r >= R_HIGH) & (ok.ratio >= RATIO_BAD)]

    lines = [
        "# e12 / /",
        "",
        f"{len(cells)} {len(ok)}",
        f"pearson_r ≥ {R_HIGH} ratio ≥ {RATIO_BAD}",
        "",
        "##",
        "",
        f"**{len(dec)} / {len(ok)} {100 * len(dec) / max(len(ok), 1):.1f}%"
        f"r ≥1 **",
        "",
    ]
    if len(dec):
        lines += [
            "Fooladi model",
            "modeltemporal****",
            "",
            "15 r",
            "",
            "| target | | | metrics | pearson_r | win_prob | | random | | |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, x in dec.sort_values("pearson_r", ascending=False).head(15).iterrows():
            lines.append(
                f"| {x['target']} | {x['pool']} | {x['selector']} | {x['metric']} | "
                f"{x['pearson_r']:.3f} | {x['win_prob']:.2f} | {x['regret']:.4f} | "
                f"{x['random_pick_regret']:.4f} | {x['ratio']:.2f} | {x['top_gap']:.4f} |"
            )
    else:
        lines += [
            "⚠ **no ** docstring",
            "datano",
            "Fooladi",
        ]

    
    lines += ["", "## r prediction", ""]
    for pool in sorted(ok["pool"].unique()):
        g = ok[ok["pool"] == pool]
        if len(g) > 3 and g.pearson_r.std() > 0 and g.ratio.std() > 0:
            rr = stats.pearsonr(g.pearson_r, g.ratio)
            sr = stats.spearmanr(g.pearson_r, g.ratio)
            lines.append(
                f"- pool={pool} n={len(g)} r Pearson"
                f"{rr.statistic:+.3f} (p={rr.pvalue:.3g})，Spearman {sr.statistic:+.3f}"
            )
    lines += [
        "",
        "message",
        "- ** 0** r",
        "- **** r r",
        "r r",
        "",
        "## Yu", "",
    ]
    g2 = ok.dropna(subset=["win_prob"])
    if len(g2) > 3 and g2.win_prob.std() > 0 and g2.ratio.std() > 0:
        rr = stats.pearsonr(g2.win_prob, g2.ratio)
        sr = stats.spearmanr(g2.win_prob, g2.ratio)
        lines.append(f"- n={len(g2)} Pearson {rr.statistic:+.3f}"
                     f"(p={rr.pvalue:.3g})，Spearman {sr.statistic:+.3f}")
        hi = g2[(g2.win_prob >= 0.75) & (g2.ratio >= RATIO_BAD)]
        lines.append(f"- ≥0.75 ≥1 **{len(hi)} / {len(g2)}**"
                     "Yu")

    
    
    
    lines += ["", "##", "",
              "= Yu", "",
              "| | | n | min | | max | |", "|---|---|---|---|---|---|---|"]
    for pool in sorted(ok["pool"].unique()):
        for wp, s in ok[ok["pool"] == pool].groupby("win_prob"):
            if len(s) >= 3:
                lines.append(
                    f"| {pool} | {wp:.2f} | {len(s)} | {s.ratio.min():.2f} | "
                    f"{s.ratio.median():.2f} | {s.ratio.max():.2f} | "
                    f"{s.ratio.max() - s.ratio.min():.2f} |")
    w1 = ok[ok.win_prob == 1.0]
    lines += [
        "",
        f"=1 {len(w1)} 0"
        f"{int((w1.ratio > 1e-9).sum())} ****"
        "temporal 0"
        "message",
        "",
        "## metrics r", "",
        "| | metrics | r median | median | ≥1 |", "|---|---|---|---|---|",
    ]
    for pool in sorted(ok["pool"].unique()):
        for m, s in ok[ok["pool"] == pool].groupby("metric"):
            lines.append(f"| {pool} | {m} | {s.pearson_r.median():.3f} | "
                         f"{s.ratio.median():.2f} | {100 * (s.ratio >= RATIO_BAD).mean():.0f}% |")

    lines += [
        "",
        "## group median", "",
        "| | | metrics | n | r median | median | median |",
        "|---|---|---|---|---|---|---|",
    ]
    grp = ok.groupby(["pool", "selector", "metric"])
    for (p, s, m), g in grp:
        lines.append(
            f"| {p} | {s} | {m} | {len(g)} | {g.pearson_r.median():.3f} | "
            f"{g.win_prob.median():.2f} | {g.ratio.median():.2f} |"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if not (E01_DIR / "e01_regret.csv").exists():
        raise SystemExit(f"e01 {E01_DIR}")
    cells = build_cells()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_DIR / "cells.csv", index=False)
    write_verdict(cells)
    print(f"\nwrite {OUT_DIR}/cells.csv verdict.md")


if __name__ == "__main__":
    main()
