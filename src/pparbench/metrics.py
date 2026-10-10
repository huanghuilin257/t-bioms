"""Documentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, pct: float) -> float:
    """Documentation."""
    n = len(y_true)
    k = max(1, int(round(n * pct / 100.0)))
    if n < 2:
        return float("nan")
    top_pred = set(np.argsort(-y_pred)[:k].tolist())
    top_true = set(np.argsort(-y_true)[:k].tolist())
    hit_rate = len(top_pred & top_true) / k
    return float(hit_rate / (k / n))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    enrichment_pct: list[float] | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, float] = {"n": int(len(y_true))}
    if len(y_true) < 3 or np.all(y_pred == y_pred[0]):
        
        
        out.update(spearman=np.nan, pearson=np.nan,
                   rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
    else:
        out.update(
            spearman=float(spearmanr(y_true, y_pred).statistic),
            pearson=float(pearsonr(y_true, y_pred).statistic),
            rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        )
    
    out["pred_std"] = float(np.std(y_pred))
    for pct in (enrichment_pct or []):
        out[f"enrich{int(pct)}"] = enrichment_factor(y_true, y_pred, pct)
    return out


def aggregate_folds(
    per_fold: list[dict],
    pooled_true: np.ndarray,
    pooled_pred: np.ndarray,
    *,
    enrichment_pct: list[float] | None = None,
) -> dict[str, float]:
    """Documentation."""
    sp = [d["spearman"] for d in per_fold if not np.isnan(d.get("spearman", np.nan))]
    pooled = compute_metrics(pooled_true, pooled_pred, enrichment_pct=enrichment_pct)
    return {
        "spearman_foldmean": float(np.mean(sp)) if sp else np.nan,
        "spearman_foldstd": float(np.std(sp)) if sp else np.nan,
        "spearman_pooled": pooled["spearman"],
        "pearson_pooled": pooled["pearson"],
        "rmse_pooled": pooled["rmse"],
        "n_folds": len(per_fold),
        "n_total": int(len(pooled_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 66,
) -> dict[str, float]:
    """Documentation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    stats = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 3 or len(np.unique(y_pred[idx])) < 3:
            continue
        stats[b] = spearmanr(y_true[idx], y_pred[idx]).statistic
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return {"spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot_ok": 0}
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot_ok": int(len(stats)),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 66,
) -> dict[str, float]:
    """Documentation."""
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    deltas = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 3:
            continue
        pa, pb = pred_a[idx], pred_b[idx]
        if len(np.unique(pa)) < 3 or len(np.unique(pb)) < 3:
            continue
        deltas[i] = spearmanr(yt, pa).statistic - spearmanr(yt, pb).statistic
    deltas = deltas[~np.isnan(deltas)]

    sp_a = float(spearmanr(y_true, pred_a).statistic)
    sp_b = float(spearmanr(y_true, pred_b).statistic)
    if len(deltas) == 0:
        return {"spearman_a": sp_a, "spearman_b": sp_b, "delta": sp_a - sp_b,
                "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                "significant": False, "n_boot_ok": 0}

    lo, hi = np.percentile(deltas, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "spearman_a": sp_a,
        "spearman_b": sp_b,
        "delta": sp_a - sp_b,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot_ok": int(len(deltas)),
    }


def summarize(df: pd.DataFrame, by: list[str], value: str = "spearman") -> pd.DataFrame:
    """Documentation."""
    g = df.groupby(by, dropna=False)[value]
    return (
        pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()})
        .reset_index()
        .sort_values("mean", ascending=False)
    )
