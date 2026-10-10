"""Documentation."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
except ImportError:      # pragma: no cover
    LGBMRegressor = None


MODEL_FAMILY = {
    "lightgbm": "boosting",
    "hgb": "boosting",
    "rf": "bagging",
    "extratrees": "bagging",
    "ridge": "reference",
    "svr": "reference",
    "knn": "reference",
}


FAMILY_CONTRAST = ("bagging", "boosting")


def _take_fp_bits(X: np.ndarray, n_bits: int = 2048) -> np.ndarray:
    """Documentation."""
    return (np.asarray(X)[:, :n_bits] > 0)


def build_models(
    cfg: dict[str, dict[str, Any]],
    *,
    seed: int,
    n_jobs: int,
    boosting_n_jobs: int = 1,
    fp_bits: int = 2048,
) -> dict[str, Any]:
    """Documentation."""
    out: dict[str, Any] = {}

    for name, params in cfg.items():
        p = dict(params or {})

        if name == "lightgbm":
            if LGBMRegressor is None:
                print("[models] warning: lightgbm skipmodel results")
                continue
            out[name] = LGBMRegressor(random_state=seed, n_jobs=boosting_n_jobs,
                                      verbose=-1, **p)

        elif name == "hgb":
            out[name] = HistGradientBoostingRegressor(random_state=seed, **p)

        elif name == "rf":
            out[name] = RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p)

        elif name == "extratrees":
            out[name] = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **p)

        elif name == "ridge":
            out[name] = make_pipeline(StandardScaler(), Ridge(random_state=seed, **p))

        elif name == "svr":
            out[name] = make_pipeline(StandardScaler(), SVR(**p))

        elif name == "knn":
            out[name] = make_pipeline(
                FunctionTransformer(_take_fp_bits, kw_args={"n_bits": fp_bits},
                                    validate=False),
                KNeighborsRegressor(metric="jaccard", algorithm="brute",
                                    n_jobs=n_jobs, **p),
            )

        else:
            raise ValueError(
                f"protocol.yaml models unknownmodel '{name}'"
                f"supported: {sorted(MODEL_FAMILY)} model MODEL_FAMILY"
            )

    if not out:
        raise RuntimeError("no usable models protocol.yaml models lightgbm")
    return out


def family_of(model_name: str) -> str:
    return MODEL_FAMILY.get(model_name, "unknown")

