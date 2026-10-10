"""Documentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs


def nearest_neighbour_similarity(
    fps: list,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Documentation."""
    train_fps = [fps[i] for i in train_idx if fps[i] is not None]
    if not train_fps:
        return np.full(len(test_idx), np.nan)
    out = np.full(len(test_idx), np.nan)
    for k, i in enumerate(test_idx):
        if fps[i] is None:
            continue
        out[k] = max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
    return out


def bucket_label(lo: float, hi: float) -> str:
    """Documentation."""
    if lo <= 0:
        return f"<{hi:g}"
    if hi > 1.0:
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def bucket_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sims: np.ndarray,
    buckets: list[list[float]],
    *,
    min_n: int = 5,
) -> pd.DataFrame:
    """Documentation."""
    from .metrics import compute_metrics

    rows = []
    for lo, hi in buckets:
        m = (sims >= lo) & (sims < hi)
        label = bucket_label(lo, hi)
        n = int(m.sum())
        if n < min_n:
            rows.append({"bucket": label, "n": n, "spearman": np.nan,
                         "rmse": np.nan, "mean_sim": np.nan, "y_std": np.nan,
                         "y_range": np.nan, "note": f"samples (<{min_n})"})
            continue
        met = compute_metrics(y_true[m], y_pred[m])
        yb = y_true[m]
        rows.append({
            "bucket": label,
            "n": n,
            "spearman": met["spearman"],
            "rmse": met["rmse"],
            "mean_sim": float(np.nanmean(sims[m])),
            "y_std": float(np.std(yb)),
            "y_range": float(np.ptp(yb)),
            "note": "",
        })
    return pd.DataFrame(rows)


def coverage(sims: np.ndarray, threshold: float = 0.4) -> dict[str, float]:
    """Documentation."""
    valid = sims[~np.isnan(sims)]
    if len(valid) == 0:
        return {"n": 0, "mean_nn_sim": np.nan,
                "in_domain_frac": np.nan, "out_of_domain_frac": np.nan}
    in_frac = float((valid >= threshold).mean())
    return {
        "n": int(len(valid)),
        "mean_nn_sim": float(valid.mean()),
        "in_domain_frac": in_frac,
        "out_of_domain_frac": 1.0 - in_frac,
    }


def monotonicity_check(bucket_df: pd.DataFrame) -> dict[str, object]:
    """Documentation."""
    sub = bucket_df.dropna(subset=["spearman"])
    
    order = {"<": 0, "0": 1, ">": 2}
    sub = sub.assign(_k=sub["bucket"].str[0].map(order)).sort_values("_k")
    vals = sub["spearman"].tolist()
    return {
        "n_buckets": len(vals),
        "values": vals,
        "monotone_increasing": bool(all(a < b for a, b in zip(vals, vals[1:]))) if len(vals) > 1 else False,
        "endpoints_increasing": bool(vals[-1] > vals[0]) if len(vals) > 1 else False,
        "endpoint_gap": float(vals[-1] - vals[0]) if len(vals) > 1 else np.nan,
    }
